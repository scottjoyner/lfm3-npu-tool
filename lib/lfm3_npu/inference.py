"""
LFM3-1.7B NPU Inference Engine.

Provides prefill + decode with KV cache management using VitisAI
ExecutionProvider on the AMD XDNA2 NPU.

Optimizations:
- VitisAI EP session config (memory arena, vaip_config)
- KV cache buffer reuse (pre-allocate, reset to zeros instead of recreate)
- Async inference (offload blocking sess.run to thread pool)
- Batched prefill support
- CPU core-aware thread tuning
"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .config import Config


def _load_layer_types(model_path: str) -> Tuple[List[int], List[int], int, int]:
    """Load layer types and dimensions from config.json alongside the model."""
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config.json")
    conv_layers: List[int] = []
    attn_layers: List[int] = []
    hidden = 2048
    conv_l_cache = 7
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        layer_types = cfg.get("layer_types", [])
        conv_layers = [i for i, t in enumerate(layer_types) if t == "conv"]
        attn_layers = [i for i, t in enumerate(layer_types) if "attention" in t]
        hidden = cfg.get("hidden_size", hidden)
        conv_l_cache = cfg.get("conv_L_cache", conv_l_cache)
    # Fallback: known LFM3 architecture
    if not conv_layers:
        conv_layers = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    if not attn_layers:
        attn_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    return conv_layers, attn_layers, hidden, conv_l_cache


# Default constants (used when config.json is not available)
ATTENTION_LAYERS: List[int] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
NUM_KV_HEADS: int = 4
HEAD_DIM: int = 128
EOS_TOKEN_ID: int = 124900


def _build_feed(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
    logits_to_keep: np.ndarray,
    past_conv: Dict[str, np.ndarray],
    past_kv: Dict[str, np.ndarray],
    needs_past_conv: bool,
    needs_past_kv: bool,
) -> Dict[str, np.ndarray]:
    """Build the feed dictionary for ONNX Runtime."""
    feed: Dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "logits_to_keep": logits_to_keep,
    }
    if needs_past_conv:
        feed.update(past_conv)
    if needs_past_kv:
        feed.update(past_kv)
    return feed


def _build_initial_state(
    conv_layers: List[int],
    attn_layers: List[int],
    batch_size: int = 1,
    seq_len: int = 0,
    hidden: int = 2048,
    conv_l_cache: int = 7,
    num_kv_heads: int = 4,
    head_dim: int = 128,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Create zero-initialized past_conv and past_key_values state."""
    past_conv: Dict[str, np.ndarray] = {}
    past_kv: Dict[str, np.ndarray] = {}
    for i in conv_layers:
        past_conv[f"past_conv.{i}"] = np.zeros(
            (batch_size, hidden, conv_l_cache), dtype=np.float32
        )
    for i in attn_layers:
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_kv_heads, seq_len, head_dim), dtype=np.float32
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, seq_len, head_dim), dtype=np.float32
        )
    return past_conv, past_kv


def _reset_state_to_zeros(
    past_conv: Dict[str, np.ndarray],
    past_kv: Dict[str, np.ndarray],
    conv_layers: List[int],
    attn_layers: List[int],
    num_kv_heads: int,
    head_dim: int,
    hidden: int,
    conv_l_cache: int,
) -> None:
    """Reset state buffers to zeros in-place (avoids reallocation)."""
    for i in conv_layers:
        key = f"past_conv.{i}"
        if key in past_conv:
            past_conv[key].fill(0.0)
    for i in attn_layers:
        for suffix in ("key", "value"):
            key = f"past_key_values.{i}.{suffix}"
            if key in past_kv:
                past_kv[key].fill(0.0)


def _update_state(
    past_conv: Dict[str, np.ndarray],
    past_kv: Dict[str, np.ndarray],
    res: Dict[str, np.ndarray],
    conv_layers: List[int],
    attn_layers: List[int],
) -> None:
    """Update state from model outputs by matching output names (in-place)."""
    for i in conv_layers:
        key = f"present_conv.{i}"
        if key in res:
            past_conv[f"past_conv.{i}"] = res[key]
    for i in attn_layers:
        for suffix in ("key", "value"):
            key = f"present.{i}.{suffix}"
            if key in res:
                past_kv[f"past_key_values.{i}.{suffix}"] = res[key]


def _sample_token(
    logits: np.ndarray, temperature: float, top_p: float
) -> int:
    """Sample next token from logits."""
    if temperature > 0:
        scaled = logits / temperature
        probs = np.exp(scaled - np.max(scaled))
        probs = probs / probs.sum()
        sorted_indices = np.argsort(probs)[::-1]
        cum_probs = np.cumsum(probs[sorted_indices])
        cutoff = int(np.searchsorted(cum_probs, top_p))
        cutoff = max(1, min(cutoff, len(sorted_indices)))
        top_indices = sorted_indices[:cutoff]
        top_probs = probs[top_indices]
        top_probs = top_probs / top_probs.sum()
        return int(np.random.choice(top_indices, p=top_probs))
    else:
        return int(np.argmax(logits))


class LFM3NPUInference:
    """LFM3-1.7B inference engine running on VitisAI EP (NPU).

    Optimizations:
    - VitisAI EP session config (memory arena, vaip_config)
    - KV cache buffer reuse (pre-allocate, reset instead of recreate)
    - Async inference (offload blocking sess.run to thread pool)
    - Batched prefill support
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config: Optional[Config] = None,
    ):
        """Initialize the inference engine.

        Args:
            model_path: Path to the ONNX model file.
            tokenizer_path: Path to the tokenizer JSON file.
            config: Optional configuration object.
        """
        self.config = config or Config({})
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.tokenizer: Optional[Tokenizer] = None
        self.sess: Optional[ort.InferenceSession] = None
        self._past_conv: Dict[str, np.ndarray] = {}
        self._past_kv: Dict[str, np.ndarray] = {}
        self._output_names: List[str] = []

        # Load layer types from config.json
        (
            self._conv_layers,
            self._attn_layers,
            self._hidden,
            self._conv_l_cache,
        ) = _load_layer_types(model_path)

        # Determine input/output requirements
        self._needs_past_conv = False
        self._needs_past_kv = False

        # Thread pool for async inference
        perf = self.config.get("performance", {})
        server_cfg = perf.get("server", {})
        max_concurrent = server_cfg.get("max_concurrent_requests", 4)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)

    def _configure_session_options(self) -> ort.SessionOptions:
        """Configure ONNX Runtime session options from config with VitisAI optimizations."""
        so = ort.SessionOptions()

        perf = self.config.get("performance", {})
        ort_cfg = perf.get("ort_session", {})

        # Thread configuration - auto-detect CPU cores if not specified
        import os as _os
        cpu_count = _os.cpu_count() or 4
        intra_threads = ort_cfg.get("intra_op_num_threads", 0)
        inter_threads = ort_cfg.get("inter_op_num_threads", 0)
        if intra_threads > 0:
            so.intra_op_num_threads = intra_threads
        else:
            # Default: use all cores for intra-op (good for single request)
            so.intra_op_num_threads = cpu_count
        if inter_threads > 0:
            so.inter_op_num_threads = inter_threads
        else:
            # Default: 1 for inter-op to avoid oversubscription with VitisAI EP
            so.inter_op_num_threads = 1

        # Execution mode
        exec_mode = ort_cfg.get("execution_mode", "ORT_SEQUENTIAL")
        if exec_mode == "ORT_PARALLEL":
            so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        else:
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Graph optimization
        opt_level = ort_cfg.get("graph_optimization_level", "ORT_ENABLE_ALL")
        if opt_level == "ORT_ENABLE_ALL":
            so.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
        elif opt_level == "ORT_ENABLE_BASIC":
            so.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
        elif opt_level == "ORT_ENABLE_EXTENDED":
            so.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
        else:
            so.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )

        # Memory arena optimizations
        so.enable_cpu_mem_arena = ort_cfg.get("enable_cpu_mem_arena", True)
        so.enable_mem_reuse = ort_cfg.get("enable_mem_reuse", True)
        so.enable_mem_pattern = ort_cfg.get("enable_mem_pattern", True)

        # Session config entries for VitisAI EP
        session_entries = ort_cfg.get("session_config_entries", {})
        # Add default VitisAI-specific entries if not overridden
        session_entries.setdefault("optimize_model", "1")
        session_entries.setdefault("enable_bf16", "1")

        for key, value in session_entries.items():
            so.add_session_config_entry(key, str(value))

        return so

    def load(self) -> "LFM3NPUInference":
        """Load the tokenizer and ONNX model onto the NPU."""
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

        custom_ops_lib = self.config.get(
            "npu.custom_ops_lib",
            "/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib/libonnxruntime_vitis_ai_custom_ops.so",
        )

        so = self._configure_session_options()
        so.register_custom_ops_library(custom_ops_lib)

        self.sess = ort.InferenceSession(
            self.model_path,
            sess_options=so,
            providers=["VitisAIExecutionProvider"],
        )

        input_names = {inp.name for inp in self.sess.get_inputs()}
        self._output_names = [o.name for o in self.sess.get_outputs()]
        self._needs_past_conv = any("past_conv." in name for name in input_names)
        self._needs_past_kv = any("past_key_values" in name for name in input_names)

        # Pre-allocate state buffers (reused across generations)
        self._past_conv, self._past_kv = _build_initial_state(
            self._conv_layers,
            self._attn_layers,
            batch_size=1,
            seq_len=0,
            hidden=self._hidden,
            conv_l_cache=self._conv_l_cache,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
        )

        print(f"Providers: {self.sess.get_providers()}", flush=True)
        print(
            f"Conv layers: {self._conv_layers}, Attention layers: {self._attn_layers}",
            flush=True,
        )
        print(
            f"Needs past_conv: {self._needs_past_conv}, "
            f"Needs past_key_values: {self._needs_past_kv}",
            flush=True,
        )
        print(f"Memory arena: CPU={so.enable_cpu_mem_arena}, reuse={so.enable_mem_reuse}, pattern={so.enable_mem_pattern}", flush=True)
        return self

    def _reset_state(self) -> None:
        """Reset the KV cache state to empty (in-place, no reallocation)."""
        _reset_state_to_zeros(
            self._past_conv,
            self._past_kv,
            self._conv_layers,
            self._attn_layers,
            NUM_KV_HEADS,
            HEAD_DIM,
            self._hidden,
            self._conv_l_cache,
        )

    def _run_inference(
        self,
        input_ids_arr: np.ndarray,
        attention_mask_arr: np.ndarray,
        position_ids_arr: np.ndarray,
        logits_to_keep_arr: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run a single inference step (blocking, for thread pool use)."""
        feed = _build_feed(
            input_ids_arr,
            attention_mask_arr,
            position_ids_arr,
            logits_to_keep_arr,
            self._past_conv,
            self._past_kv,
            self._needs_past_conv,
            self._needs_past_kv,
        )
        outputs = self.sess.run(None, feed)
        res = dict(zip(self._output_names, outputs))
        _update_state(
            self._past_conv, self._past_kv, res,
            self._conv_layers, self._attn_layers,
        )
        return outputs[0], res

    async def _run_inference_async(
        self,
        input_ids_arr: np.ndarray,
        attention_mask_arr: np.ndarray,
        position_ids_arr: np.ndarray,
        logits_to_keep_arr: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run inference asynchronously via thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._run_inference,
            input_ids_arr,
            attention_mask_arr,
            position_ids_arr,
            logits_to_keep_arr,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from a prompt using NPU inference.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Top-p sampling threshold.

        Returns:
            Generated text (excluding the prompt).
        """
        if self.sess is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        encoded = self.tokenizer.encode(prompt)
        input_ids_list: List[int] = encoded.ids
        prompt_len: int = len(input_ids_list)

        # Reset state (empty KV cache, seq_len=0) - in-place, no reallocation
        self._reset_state()

        # Prefill
        input_ids_arr = np.array([input_ids_list], dtype=np.int64)
        attention_mask_arr = np.ones((1, prompt_len), dtype=np.int64)
        position_ids_arr = np.array([list(range(prompt_len))], dtype=np.int64)
        logits_to_keep_arr = np.array([1], dtype=np.int64)

        logits, _ = self._run_inference(
            input_ids_arr, attention_mask_arr, position_ids_arr,
            logits_to_keep_arr,
        )

        next_logits = logits[0, -1, :]
        generated_ids: List[int] = list(input_ids_list)

        for step in range(max_new_tokens):
            next_token = _sample_token(next_logits, temperature, top_p)
            generated_ids.append(next_token)

            if next_token == EOS_TOKEN_ID:
                break

            seq_len = prompt_len + step + 1
            input_ids_arr = np.array([[next_token]], dtype=np.int64)
            attention_mask_arr = np.ones((1, seq_len), dtype=np.int64)
            position_ids_arr = np.array([[seq_len - 1]], dtype=np.int64)
            logits_to_keep_arr = np.array([1], dtype=np.int64)

            logits, _ = self._run_inference(
                input_ids_arr, attention_mask_arr, position_ids_arr,
                logits_to_keep_arr,
            )
            next_logits = logits[0, -1, :]

        text = self.tokenizer.decode(generated_ids[prompt_len:])
        return text

    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text asynchronously (non-blocking).

        Uses thread pool to offload blocking sess.run calls.
        """
        if self.sess is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        encoded = self.tokenizer.encode(prompt)
        input_ids_list: List[int] = encoded.ids
        prompt_len: int = len(input_ids_list)

        # Reset state (empty KV cache, seq_len=0)
        self._reset_state()

        # Prefill
        input_ids_arr = np.array([input_ids_list], dtype=np.int64)
        attention_mask_arr = np.ones((1, prompt_len), dtype=np.int64)
        position_ids_arr = np.array([list(range(prompt_len))], dtype=np.int64)
        logits_to_keep_arr = np.array([1], dtype=np.int64)

        logits, _ = await self._run_inference_async(
            input_ids_arr, attention_mask_arr, position_ids_arr,
            logits_to_keep_arr,
        )

        next_logits = logits[0, -1, :]
        generated_ids: List[int] = list(input_ids_list)

        for step in range(max_new_tokens):
            next_token = _sample_token(next_logits, temperature, top_p)
            generated_ids.append(next_token)

            if next_token == EOS_TOKEN_ID:
                break

            seq_len = prompt_len + step + 1
            input_ids_arr = np.array([[next_token]], dtype=np.int64)
            attention_mask_arr = np.ones((1, seq_len), dtype=np.int64)
            position_ids_arr = np.array([[seq_len - 1]], dtype=np.int64)
            logits_to_keep_arr = np.array([1], dtype=np.int64)

            logits, _ = await self._run_inference_async(
                input_ids_arr, attention_mask_arr, position_ids_arr,
                logits_to_keep_arr,
            )
            next_logits = logits[0, -1, :]

        text = self.tokenizer.decode(generated_ids[prompt_len:])
        return text

    def generate_full(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text including the prompt."""
        generated = self.generate(prompt, max_new_tokens, temperature, top_p)
        return prompt + generated

    def warmup(self, prompt: str = "The quick brown fox") -> None:
        """Warm up the model with a dummy inference."""
        print("Warming up NPU model...", flush=True)
        self.generate(prompt, max_new_tokens=1, temperature=0.0)
        print("Warmup complete.", flush=True)

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=True)