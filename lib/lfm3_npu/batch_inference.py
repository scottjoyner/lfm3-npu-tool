"""
Continuous batching inference engine for LFM3-1.7B.

vLLM-style continuous batching:
- Batched prefill: multiple prompts in a single sess.run() (all start with empty KV cache)
- Batched decode with padding: multiple active requests decoded together in a single
  sess.run(), padding shorter KV caches to match the longest active request
- Single-request fast path: when only 1 pending request and 0 active requests,
  bypass the batch loop and call sess.run() directly for minimum latency
- Per-request KV cache slots within pre-allocated batch tensors
- Dynamic entry/exit: requests join on arrival, leave when finished

Optimizations:
- Pre-allocated batch tensors with auto-growth as sequences grow
- KV cache buffer reuse (in-place reset, never shrink)
- VitisAI EP memory arena + session config
- Async inference offloading to thread pool
- Tokenizer LRU cache for repeated prompts
- Single-request fast path for minimum latency
- Batched decode with padding for maximum throughput under concurrency
- Per-request metrics (TTFT, TPS, token times)
"""

import asyncio
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .config import Config


def _load_layer_types(model_path: str) -> Tuple[List[int], List[int], int, int]:
    """Load layer types and dimensions from config.json alongside the model."""
    import json as _json

    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config.json")
    conv_layers: List[int] = []
    attn_layers: List[int] = []
    hidden = 2048
    conv_l_cache = 7
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = _json.load(f)
        layer_types = cfg.get("layer_types", [])
        conv_layers = [i for i, t in enumerate(layer_types) if t == "conv"]
        attn_layers = [i for i, t in enumerate(layer_types) if "attention" in t]
        hidden = cfg.get("hidden_size", hidden)
        conv_l_cache = cfg.get("conv_L_cache", conv_l_cache)
    if not conv_layers:
        conv_layers = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    if not attn_layers:
        attn_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    return conv_layers, attn_layers, hidden, conv_l_cache


ATTENTION_LAYERS: List[int] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
NUM_KV_HEADS: int = 4
HEAD_DIM: int = 128
EOS_TOKEN_ID: int = 124900


def _sample_token(logits, temperature, top_p):
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


class ReqState:
    """Per-request generation state within a batch slot."""

    def __init__(self, req_id: str, prompt: str, input_ids: List[int], slot: int):
        self.req_id = req_id
        self.prompt = prompt
        self.input_ids = input_ids
        self.prompt_len = len(input_ids)
        self.generated_ids: List[int] = list(input_ids)
        self.next_token: Optional[int] = None
        self.slot = slot
        self.finished = False
        self.finished_reason: Optional[str] = None
        self.prefill_start: Optional[float] = None
        self.prefill_done = False
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_tokens = 50
        self.token_times: List[float] = []
        self.kv_len = 0  # Track KV cache length explicitly
        self.first_token_time: Optional[float] = None
        self.queue_wait_start: Optional[float] = None


class TokenizerCache:
    """LRU cache for tokenizer.encode results."""

    def __init__(self, maxsize: int = 1000):
        self.cache: OrderedDict[str, List[int]] = OrderedDict()
        self.maxsize = maxsize

    def encode(self, tokenizer: Tokenizer, text: str) -> List[int]:
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        ids = tokenizer.encode(text).ids
        if len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[text] = ids
        return ids


class ContinuousBatcher:
    """
    Continuous batching manager for LFM3 inference.

    Uses batched prefill + batched decode with padding to avoid KV cache sequence
    length mismatches that occur with the ONNX model's GroupQueryAttention operator.
    Includes a single-request fast path for minimum latency.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config: Optional[Config] = None,
        max_batch_size: int = 8,
    ):
        self.config = config or Config({})
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_batch_size = max_batch_size

        (
            self._conv_layers,
            self._attn_layers,
            self._hidden,
            self._conv_l_cache,
        ) = _load_layer_types(model_path)

        self._needs_past_conv = False
        self._needs_past_kv = False

        self.tokenizer: Optional[Tokenizer] = None
        self._tokenizer_cache = TokenizerCache(maxsize=500)
        self.sess: Optional[ort.InferenceSession] = None
        self._output_names: List[str] = []

        self._slots: List[Optional[ReqState]] = [None] * max_batch_size
        self._free_slots: List[int] = list(range(max_batch_size))

        self._batch_past_conv: Dict[str, np.ndarray] = {}
        self._batch_past_kv: Dict[str, np.ndarray] = {}

        perf = self.config.get("performance", {})
        server_cfg = perf.get("server", {})
        max_concurrent = server_cfg.get("max_concurrent_requests", max_batch_size)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)

        self._total_prefill_time = 0.0
        self._total_decode_time = 0.0
        self._prefill_count = 0
        self._decode_count = 0
        self._total_tokens_generated = 0
        self._fast_path_count = 0
        self._batched_decode_count = 0

    # ------------------------------------------------------------------ #
    # Session configuration
    # ------------------------------------------------------------------ #
    def _configure_session_options(self, provider: str = "CPUExecutionProvider") -> ort.SessionOptions:
        """Configure ONNX Runtime session options with VitisAI optimizations."""
        so = ort.SessionOptions()

        perf = self.config.get("performance", {})
        ort_cfg = perf.get("ort_session", {})

        import os as _os
        cpu_count = _os.cpu_count() or 4
        intra_threads = ort_cfg.get("intra_op_num_threads", 0)
        inter_threads = ort_cfg.get("inter_op_num_threads", 0)
        if intra_threads > 0:
            so.intra_op_num_threads = intra_threads
        else:
            so.intra_op_num_threads = cpu_count
        if inter_threads > 0:
            so.inter_op_num_threads = inter_threads
        else:
            so.inter_op_num_threads = 1

        # Execution mode: ORT_PARALLEL for CPU (better prefill parallelism),
        # ORT_SEQUENTIAL for VitisAI EP (NPU handles its own parallelism)
        exec_mode = ort_cfg.get("execution_mode", "ORT_SEQUENTIAL")
        if provider == "CPUExecutionProvider" and exec_mode == "ORT_SEQUENTIAL":
            # Override to PARALLEL for CPU EP for better prefill performance
            so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        else:
            so.execution_mode = (
                ort.ExecutionMode.ORT_PARALLEL
                if exec_mode == "ORT_PARALLEL"
                else ort.ExecutionMode.ORT_SEQUENTIAL
            )

        opt_level = ort_cfg.get("graph_optimization_level", "ORT_ENABLE_ALL")
        level_map = {
            "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        }
        so.graph_optimization_level = level_map.get(
            opt_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        so.enable_cpu_mem_arena = ort_cfg.get("enable_cpu_mem_arena", True)
        so.enable_mem_reuse = ort_cfg.get("enable_mem_reuse", True)
        so.enable_mem_pattern = ort_cfg.get("enable_mem_pattern", True)

        session_entries = ort_cfg.get("session_config_entries", {})
        session_entries.setdefault("optimize_model", "1")
        session_entries.setdefault("enable_bf16", "1")
        for key, value in session_entries.items():
            so.add_session_config_entry(key, str(value))

        return so

    # ------------------------------------------------------------------ #
    # Load / shutdown
    # ------------------------------------------------------------------ #
    def load(self, provider: str = "CPUExecutionProvider") -> "ContinuousBatcher":
        """Load the tokenizer and ONNX model, pre-allocate batch state."""
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

        custom_ops_lib = self.config.get(
            "npu.custom_ops_lib",
            "/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib/libonnxruntime_vitis_ai_custom_ops.so",
        )

        so = self._configure_session_options(provider)
        
        # Only register custom ops for VitisAI EP
        if provider == "VitisAIExecutionProvider":
            so.register_custom_ops_library(custom_ops_lib)

        self.sess = ort.InferenceSession(
            self.model_path,
            sess_options=so,
            providers=[provider],
        )

        input_names = {inp.name for inp in self.sess.get_inputs()}
        self._output_names = [o.name for o in self.sess.get_outputs()]
        self._needs_past_conv = any("past_conv." in name for name in input_names)
        self._needs_past_kv = any("past_key_values" in name for name in input_names)

        # Pre-allocate batch state tensors (batch_size = max_batch_size, seq_len=0)
        batch = self.max_batch_size
        if self._needs_past_conv:
            for i in self._conv_layers:
                self._batch_past_conv[f"past_conv.{i}"] = np.zeros(
                    (batch, self._hidden, self._conv_l_cache), dtype=np.float32
                )
        if self._needs_past_kv:
            for i in self._attn_layers:
                self._batch_past_kv[f"past_key_values.{i}.key"] = np.zeros(
                    (batch, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32
                )
                self._batch_past_kv[f"past_key_values.{i}.value"] = np.zeros(
                    (batch, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32
                )

        print(
            f"ContinuousBatcher: Providers: {self.sess.get_providers()}", flush=True
        )
        print(
            f"ContinuousBatcher: Conv layers: {self._conv_layers}, "
            f"Attention layers: {self._attn_layers}",
            flush=True,
        )
        print(
            f"ContinuousBatcher: max_batch_size={self.max_batch_size}, "
            f"needs_past_conv={self._needs_past_conv}, "
            f"needs_past_key_values={self._needs_past_kv}",
            flush=True,
        )
        print(
            f"ContinuousBatcher: Memory arena CPU={so.enable_cpu_mem_arena}, "
            f"reuse={so.enable_mem_reuse}, pattern={so.enable_mem_pattern}",
            flush=True,
        )
        return self

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=True)

    # ------------------------------------------------------------------ #
    # Warmup
    # ------------------------------------------------------------------ #
    def warmup(self, prompt: str = "The quick brown fox", max_tokens: int = 5) -> None:
        """Run a dummy inference to load model into memory and CPU caches."""
        print(f"ContinuousBatcher: Warming up with '{prompt}' ({max_tokens} tokens)...", flush=True)
        t0 = time.perf_counter()
        self.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
        t1 = time.perf_counter()
        print(f"ContinuousBatcher: Warmup complete in {(t1-t0)*1000:.0f}ms", flush=True)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Synchronous generate (for warmup and testing)."""
        input_ids = self._tokenizer_cache.encode(self.tokenizer, prompt)
        prompt_len = len(input_ids)

        # Reset slot 0
        self._reset_slot_to_zeros(0)

        # Prefill (batch_size=1, kv_len=0)
        input_ids_arr = np.array([input_ids], dtype=np.int64)
        attention_mask_arr = np.ones((1, prompt_len), dtype=np.int64)
        position_ids_arr = np.array([list(range(prompt_len))], dtype=np.int64)
        logits_to_keep_arr = np.array([1], dtype=np.int64)

        feed = self._build_feed_single(
            input_ids_arr, attention_mask_arr, position_ids_arr, logits_to_keep_arr, 0, kv_len=0
        )
        outputs = self.sess.run(None, feed)
        res = dict(zip(self._output_names, outputs))
        self._update_batch_state_single(res, 0)

        logits = outputs[0]
        next_logits = logits[0, 0, :] if logits.ndim == 3 else logits[0, -1, :]

        generated_ids = list(input_ids)
        kv_len = prompt_len
        for step in range(max_new_tokens):
            next_token = _sample_token(next_logits, temperature, top_p)
            generated_ids.append(next_token)
            if next_token == EOS_TOKEN_ID:
                break

            seq_len = kv_len + 1
            input_ids_arr = np.array([[next_token]], dtype=np.int64)
            attention_mask_arr = np.ones((1, seq_len), dtype=np.int64)
            position_ids_arr = np.array([[kv_len]], dtype=np.int64)
            logits_to_keep_arr = np.array([1], dtype=np.int64)

            feed = self._build_feed_single(
                input_ids_arr, attention_mask_arr, position_ids_arr, logits_to_keep_arr, 0, kv_len=kv_len
            )
            outputs = self.sess.run(None, feed)
            res = dict(zip(self._output_names, outputs))
            self._update_batch_state_single(res, 0)
            kv_len += 1
            logits = outputs[0]
            next_logits = logits[0, 0, :] if logits.ndim == 3 else logits[0, -1, :]

        return self.tokenizer.decode(generated_ids[prompt_len:])

    # ------------------------------------------------------------------ #
    # Batch state management (single slot)
    # ------------------------------------------------------------------ #
    def _reset_slot_to_zeros(self, slot: int) -> None:
        """Reset a single batch slot's KV cache to zeros in-place."""
        if self._needs_past_conv:
            for i in self._conv_layers:
                key = f"past_conv.{i}"
                if key in self._batch_past_conv:
                    self._batch_past_conv[key][slot] = 0.0
        if self._needs_past_kv:
            for i in self._attn_layers:
                for suffix in ("key", "value"):
                    key = f"past_key_values.{i}.{suffix}"
                    if key in self._batch_past_kv:
                        self._batch_past_kv[key][slot] = 0.0

    def _ensure_kv_shape(
        self, past_key: str, needed_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Ensure KV cache tensor has the right shape, growing if needed.

        Only grows (never shrinks) to preserve data for other batch slots.
        Takes the max of each dimension between current and needed.
        """
        current = self._batch_past_kv.get(past_key)
        if current is None:
            self._batch_past_kv[past_key] = np.zeros(needed_shape, dtype=np.float32)
        elif current.shape != needed_shape:
            new_shape = tuple(max(c, n) for c, n in zip(current.shape, needed_shape))
            if new_shape != current.shape:
                new_arr = np.zeros(new_shape, dtype=np.float32)
                copy_seq = min(current.shape[2], new_shape[2])
                new_arr[:, :, :copy_seq, :] = current[:, :, :copy_seq, :]
                self._batch_past_kv[past_key] = new_arr
        return self._batch_past_kv[past_key]

    def _build_feed_single(
        self,
        input_ids_arr: np.ndarray,
        attention_mask_arr: np.ndarray,
        position_ids_arr: np.ndarray,
        logits_to_keep_arr: np.ndarray,
        slot: int,
        kv_len: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Build feed dict for a single slot (used by decode).

        Slices KV cache to current valid length (kv_len) so the ONNX model
        sees the correct past_sequence_length.
        """
        feed: Dict[str, np.ndarray] = {
            "input_ids": input_ids_arr,
            "attention_mask": attention_mask_arr,
            "position_ids": position_ids_arr,
            "logits_to_keep": logits_to_keep_arr,
        }

        if self._needs_past_conv:
            for i in self._conv_layers:
                key = f"past_conv.{i}"
                if key in self._batch_past_conv:
                    feed[key] = self._batch_past_conv[key][slot : slot + 1]
        if self._needs_past_kv:
            for i in self._attn_layers:
                for suffix in ("key", "value"):
                    key = f"past_key_values.{i}.{suffix}"
                    if key in self._batch_past_kv:
                        feed[key] = self._batch_past_kv[key][slot : slot + 1, :, :kv_len, :]
        return feed

    def _build_feed(
        self,
        input_ids_arr: np.ndarray,
        attention_mask_arr: np.ndarray,
        position_ids_arr: np.ndarray,
        logits_to_keep_arr: np.ndarray,
        active_slots: List[int],
    ) -> Dict[str, np.ndarray]:
        """Build feed dict for sess.run() with active slot selection (batched prefill).

        For prefill, all slots start with empty KV cache (kv_len=0).
        """
        feed: Dict[str, np.ndarray] = {
            "input_ids": input_ids_arr,
            "attention_mask": attention_mask_arr,
            "position_ids": position_ids_arr,
            "logits_to_keep": logits_to_keep_arr,
        }

        if self._needs_past_conv:
            for i in self._conv_layers:
                key = f"past_conv.{i}"
                if key in self._batch_past_conv:
                    feed[key] = self._batch_past_conv[key][active_slots]
        if self._needs_past_kv:
            for i in self._attn_layers:
                for suffix in ("key", "value"):
                    key = f"past_key_values.{i}.{suffix}"
                    if key in self._batch_past_kv:
                        feed[key] = self._batch_past_kv[key][active_slots, :, :0, :]
        return feed

    def _build_feed_batched_decode(
        self,
        input_ids_arr: np.ndarray,
        attention_mask_arr: np.ndarray,
        position_ids_arr: np.ndarray,
        logits_to_keep_arr: np.ndarray,
        active_slots: List[int],
        kv_lens: List[int],
        max_kv_len: int,
    ) -> Dict[str, np.ndarray]:
        """Build feed dict for batched decode with KV cache padding.

        Pads each request's KV cache to max_kv_len so the ONNX model sees
        uniform sequence length across all batch elements.
        """
        feed: Dict[str, np.ndarray] = {
            "input_ids": input_ids_arr,
            "attention_mask": attention_mask_arr,
            "position_ids": position_ids_arr,
            "logits_to_keep": logits_to_keep_arr,
        }

        if self._needs_past_conv:
            for i in self._conv_layers:
                key = f"past_conv.{i}"
                if key in self._batch_past_conv:
                    feed[key] = self._batch_past_conv[key][active_slots]
        if self._needs_past_kv:
            for i in self._attn_layers:
                for suffix in ("key", "value"):
                    key = f"past_key_values.{i}.{suffix}"
                    if key in self._batch_past_kv:
                        # Build padded KV cache: slice each slot to its kv_len,
                        # then pad with zeros to max_kv_len
                        padded = np.zeros(
                            (len(active_slots), NUM_KV_HEADS, max_kv_len, HEAD_DIM),
                            dtype=np.float32,
                        )
                        for j, slot in enumerate(active_slots):
                            kv_len = kv_lens[j]
                            if kv_len > 0:
                                padded[j, :, :kv_len, :] = self._batch_past_kv[key][
                                    slot : slot + 1, :, :kv_len, :
                                ]
                        feed[key] = padded
        return feed

    def _update_batch_state_single(
        self, res: Dict[str, np.ndarray], slot: int
    ) -> None:
        """Update batch state from model outputs for a single slot."""
        # conv layers
        if self._needs_past_conv:
            for i in self._conv_layers:
                key = f"present_conv.{i}"
                if key in res:
                    self._batch_past_conv[f"past_conv.{i}"][slot] = res[key][0]

        # attention layers
        if self._needs_past_kv:
            for i in self._attn_layers:
                for suffix in ("key", "value"):
                    key = f"present.{i}.{suffix}"
                    if key in res:
                        past_key = f"past_key_values.{i}.{suffix}"
                        out = res[key]
                        needed_shape = (self.max_batch_size,) + out.shape[1:]
                        arr = self._ensure_kv_shape(past_key, needed_shape)
                        arr[slot, :, :out.shape[2], :] = out[0]

    def _update_batch_state(
        self, res: Dict[str, np.ndarray], active_slots: List[int]
    ) -> None:
        """Update batch state from model outputs for active slots (batched prefill)."""
        num_active = len(active_slots)

        if self._needs_past_conv:
            for i in self._conv_layers:
                key = f"present_conv.{i}"
                if key in res:
                    out = res[key]
                    if out.shape[0] == num_active:
                        for j, slot in enumerate(active_slots):
                            self._batch_past_conv[f"past_conv.{i}"][slot] = out[j]
                    else:
                        self._batch_past_conv[f"past_conv.{i}"][:] = out

        if self._needs_past_kv:
            for i in self._attn_layers:
                for suffix in ("key", "value"):
                    key = f"present.{i}.{suffix}"
                    if key in res:
                        out = res[key]
                        past_key = f"past_key_values.{i}.{suffix}"
                        if out.shape[0] == num_active:
                            needed_shape = (self.max_batch_size,) + out.shape[1:]
                            arr = self._ensure_kv_shape(past_key, needed_shape)
                            for j, slot in enumerate(active_slots):
                                arr[slot, :, :out.shape[2], :] = out[j]
                        else:
                            needed_shape = out.shape
                            arr = self._ensure_kv_shape(past_key, needed_shape)
                            arr[:] = out

    def _update_batch_state_batched_decode(
        self, res: Dict[str, np.ndarray], active_slots: List[int], kv_lens: List[int]
    ) -> None:
        """Update batch state from batched decode outputs.

        Writes each slot's output up to its kv_len+1 (new token added).
        """
        if self._needs_past_conv:
            for i in self._conv_layers:
                key = f"present_conv.{i}"
                if key in res:
                    out = res[key]
                    for j, slot in enumerate(active_slots):
                        self._batch_past_conv[f"past_conv.{i}"][slot] = out[j]

        if self._needs_past_kv:
            for i in self._attn_layers:
                for suffix in ("key", "value"):
                    key = f"present.{i}.{suffix}"
                    if key in res:
                        out = res[key]
                        past_key = f"past_key_values.{i}.{suffix}"
                        needed_shape = (self.max_batch_size,) + out.shape[1:]
                        arr = self._ensure_kv_shape(past_key, needed_shape)
                        for j, slot in enumerate(active_slots):
                            new_kv_len = kv_lens[j] + 1
                            arr[slot, :, :new_kv_len, :] = out[j, :, :new_kv_len, :]

    # ------------------------------------------------------------------ #
    # Single-step inference (blocking, for thread pool)
    # ------------------------------------------------------------------ #
    def _run_batch_step(
        self,
        input_ids_arr: np.ndarray,
        attention_mask_arr: np.ndarray,
        position_ids_arr: np.ndarray,
        logits_to_keep_arr: np.ndarray,
        active_slots: List[int],
    ) -> np.ndarray:
        """Run a single batched inference step."""
        feed = self._build_feed(
            input_ids_arr, attention_mask_arr, position_ids_arr,
            logits_to_keep_arr, active_slots,
        )
        outputs = self.sess.run(None, feed)
        res = dict(zip(self._output_names, outputs))
        self._update_batch_state(res, active_slots)
        return outputs[0]

    def _run_single_step(
        self,
        input_ids_arr: np.ndarray,
        attention_mask_arr: np.ndarray,
        position_ids_arr: np.ndarray,
        logits_to_keep_arr: np.ndarray,
        slot: int,
        kv_len: int = 0,
    ) -> np.ndarray:
        """Run a single inference step for one slot."""
        feed = self._build_feed_single(
            input_ids_arr, attention_mask_arr, position_ids_arr,
            logits_to_keep_arr, slot, kv_len,
        )
        outputs = self.sess.run(None, feed)
        res = dict(zip(self._output_names, outputs))
        self._update_batch_state_single(res, slot)
        return outputs[0]

    def _run_batched_decode_step(
        self,
        input_ids_arr: np.ndarray,
        attention_mask_arr: np.ndarray,
        position_ids_arr: np.ndarray,
        logits_to_keep_arr: np.ndarray,
        active_slots: List[int],
        kv_lens: List[int],
        max_kv_len: int,
    ) -> np.ndarray:
        """Run a single batched decode step with KV cache padding."""
        feed = self._build_feed_batched_decode(
            input_ids_arr, attention_mask_arr, position_ids_arr,
            logits_to_keep_arr, active_slots, kv_lens, max_kv_len,
        )
        outputs = self.sess.run(None, feed)
        res = dict(zip(self._output_names, outputs))
        self._update_batch_state_batched_decode(res, active_slots, kv_lens)
        return outputs[0]

    async def _run_batch_step_async(
        self,
        input_ids_arr: np.ndarray,
        attention_mask_arr: np.ndarray,
        position_ids_arr: np.ndarray,
        logits_to_keep_arr: np.ndarray,
        active_slots: List[int],
    ) -> np.ndarray:
        """Run a single batched inference step asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._run_batch_step,
            input_ids_arr,
            attention_mask_arr,
            position_ids_arr,
            logits_to_keep_arr,
            active_slots,
        )

    async def _run_single_step_async(
        self,
        input_ids_arr: np.ndarray,
        attention_mask_arr: np.ndarray,
        position_ids_arr: np.ndarray,
        logits_to_keep_arr: np.ndarray,
        slot: int,
        kv_len: int = 0,
    ) -> np.ndarray:
        """Run a single inference step asynchronously for one slot."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._run_single_step,
            input_ids_arr,
            attention_mask_arr,
            position_ids_arr,
            logits_to_keep_arr,
            slot,
            kv_len,
        )

    async def _run_batched_decode_step_async(
        self,
        input_ids_arr: np.ndarray,
        attention_mask_arr: np.ndarray,
        position_ids_arr: np.ndarray,
        logits_to_keep_arr: np.ndarray,
        active_slots: List[int],
        kv_lens: List[int],
        max_kv_len: int,
    ) -> np.ndarray:
        """Run a single batched decode step asynchronously with padding."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._run_batched_decode_step,
            input_ids_arr,
            attention_mask_arr,
            position_ids_arr,
            logits_to_keep_arr,
            active_slots,
            kv_lens,
            max_kv_len,
        )

    # ------------------------------------------------------------------ #
    # Single-request fast path
    # ------------------------------------------------------------------ #
    async def fast_path_generate_single(
        self, req: ReqState
    ) -> Tuple[str, int, bool, str, float]:
        """
        Single-request fast path: prefill + decode one step without batch loop overhead.

        Used when there is exactly 1 pending request and 0 active requests.
        Returns (req_id, next_token, finished, finish_reason, ttft).
        """
        slot = req.slot
        self._reset_slot_to_zeros(slot)

        # Prefill
        input_ids_arr = np.array([req.input_ids], dtype=np.int64)
        attention_mask_arr = np.ones((1, req.prompt_len), dtype=np.int64)
        position_ids_arr = np.array([list(range(req.prompt_len))], dtype=np.int64)
        logits_to_keep_arr = np.array([1], dtype=np.int64)

        t0 = time.perf_counter()
        feed = self._build_feed_single(
            input_ids_arr, attention_mask_arr, position_ids_arr,
            logits_to_keep_arr, slot, kv_len=0,
        )
        outputs = self.sess.run(None, feed)
        res = dict(zip(self._output_names, outputs))
        self._update_batch_state_single(res, slot)
        t1 = time.perf_counter()

        self._total_prefill_time += t1 - t0
        self._prefill_count += 1

        req.kv_len = req.prompt_len
        req.prefill_start = t0
        req.prefill_done = True

        logits = outputs[0]
        next_logits = logits[0, 0, :] if logits.ndim == 3 else logits[0, -1, :]

        next_token = _sample_token(next_logits, req.temperature, req.top_p)
        req.generated_ids.append(next_token)
        req.next_token = next_token

        finished = False
        finish_reason = None
        if next_token == EOS_TOKEN_ID:
            finished = True
            finish_reason = "stop"
        elif len(req.generated_ids) - req.prompt_len >= req.max_tokens:
            finished = True
            finish_reason = "length"

        if not finished:
            req.kv_len += 1

        ttft = time.perf_counter() - t0
        self._fast_path_count += 1

        return (req.req_id, next_token, finished, finish_reason, ttft)

    def _validate_kv_len(self, slot: int, req_kv_len: int) -> int:
        """Validate and re-sync req.kv_len against actual KV cache tensor shape.

        Returns the corrected kv_len.
        """
        if not self._needs_past_kv or not self._attn_layers:
            return req_kv_len
        attn_layer = self._attn_layers[0]
        key = f"past_key_values.{attn_layer}.key"
        tensor = self._batch_past_kv.get(key)
        if tensor is None:
            return req_kv_len
        actual_len = tensor.shape[2]
        if req_kv_len != actual_len:
            # Re-sync: actual tensor length is the source of truth
            return actual_len
        return req_kv_len

    async def fast_path_decode_single(
        self, req: ReqState
    ) -> Tuple[str, int, bool, str, float]:
        """Single-request fast path decode step."""
        slot = req.slot

        # Validate kv_len against actual KV cache tensor (recover from shape drift)
        kv_seq_len = self._validate_kv_len(slot, req.kv_len)
        req.kv_len = kv_seq_len
        seq_len = kv_seq_len + 1

        input_ids_arr = np.zeros((1, 1), dtype=np.int64)
        attention_mask_arr = np.ones((1, seq_len), dtype=np.int64)
        position_ids_arr = np.array([[kv_seq_len]], dtype=np.int64)
        logits_to_keep_arr = np.array([1], dtype=np.int64)

        last_token = req.next_token if req.next_token is not None else req.input_ids[-1]
        input_ids_arr[0, 0] = last_token

        t0 = time.perf_counter()
        logits = await self._run_single_step_async(
            input_ids_arr,
            attention_mask_arr,
            position_ids_arr,
            logits_to_keep_arr,
            slot,
            kv_len=req.kv_len,
        )
        t1 = time.perf_counter()

        self._total_decode_time += t1 - t0
        self._decode_count += 1

        token_start = time.perf_counter()
        req_logits = logits[0, 0, :]

        next_token = _sample_token(req_logits, req.temperature, req.top_p)

        req.generated_ids.append(next_token)
        req.next_token = next_token
        req.kv_len += 1
        self._total_tokens_generated += 1

        finished = False
        finish_reason = None
        if next_token == EOS_TOKEN_ID:
            finished = True
            finish_reason = "stop"
        elif len(req.generated_ids) - req.prompt_len >= req.max_tokens:
            finished = True
            finish_reason = "length"

        token_duration = time.perf_counter() - token_start
        req.token_times.append(token_duration)

        return (req.req_id, next_token, finished, finish_reason, token_duration)

    # ------------------------------------------------------------------ #
    # Prefill (batched)
    # ------------------------------------------------------------------ #
    async def prefill_batch(
        self, requests: List[ReqState]
    ) -> List[Tuple[str, np.ndarray]]:
        """Prefill a batch of new requests in a single sess.run()."""
        if not requests:
            return []

        batch_size = len(requests)
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}"
            )

        slots = []
        for req in requests:
            if not self._free_slots:
                raise RuntimeError("No free batch slots available")
            slot = self._free_slots.pop(0)
            req.slot = slot
            self._slots[slot] = req
            slots.append(slot)

        for slot in slots:
            self._reset_slot_to_zeros(slot)

        max_prompt_len = max(req.prompt_len for req in requests)
        input_ids_arr = np.zeros((batch_size, max_prompt_len), dtype=np.int64)
        attention_mask_arr = np.zeros((batch_size, max_prompt_len), dtype=np.int64)
        position_ids_arr = np.zeros((batch_size, max_prompt_len), dtype=np.int64)

        for j, req in enumerate(requests):
            input_ids_arr[j, : req.prompt_len] = req.input_ids
            attention_mask_arr[j, : req.prompt_len] = 1
            position_ids_arr[j, : req.prompt_len] = list(range(req.prompt_len))

        logits_to_keep_arr = np.array([1], dtype=np.int64)

        t0 = time.perf_counter()
        logits = await self._run_batch_step_async(
            input_ids_arr,
            attention_mask_arr,
            position_ids_arr,
            logits_to_keep_arr,
            slots,
        )
        t1 = time.perf_counter()

        self._total_prefill_time += t1 - t0
        self._prefill_count += 1

        results: List[Tuple[str, np.ndarray]] = []
        for j, req in enumerate(requests):
            req.prefill_done = True
            req.prefill_start = t0
            req.kv_len = req.prompt_len
            req_logits = logits[j, 0, :]
            results.append((req.req_id, req_logits))

        return results

    # ------------------------------------------------------------------ #
    # Decode (batched with padding)
    # ------------------------------------------------------------------ #
    async def decode_batched(
        self, active_requests: List[ReqState]
    ) -> List[Tuple[str, int, bool, str, float]]:
        """
        Batched decode with KV cache padding for all active requests in one sess.run().

        Pads shorter KV caches to match the longest active request so the ONNX
        model sees uniform sequence length across all batch elements.
        """
        if not active_requests:
            return []

        batch_size = len(active_requests)
        slots = [req.slot for req in active_requests]

        # Validate and re-sync kv_len against actual KV cache tensors
        validated_kv_lens = []
        for req in active_requests:
            corrected = self._validate_kv_len(req.slot, req.kv_len)
            req.kv_len = corrected
            validated_kv_lens.append(corrected)

        # KV cache lengths tracked explicitly
        kv_lens = validated_kv_lens
        max_kv_len = max(kv_lens)
        seq_len = max_kv_len + 1  # +1 for the new token we're processing

        input_ids_arr = np.zeros((batch_size, 1), dtype=np.int64)
        attention_mask_arr = np.ones((batch_size, seq_len), dtype=np.int64)
        position_ids_arr = np.zeros((batch_size, 1), dtype=np.int64)
        logits_to_keep_arr = np.array([1], dtype=np.int64)

        for j, req in enumerate(active_requests):
            last_token = req.next_token if req.next_token is not None else req.input_ids[-1]
            input_ids_arr[j, 0] = last_token
            position_ids_arr[j, 0] = kv_lens[j]

        t0 = time.perf_counter()
        logits = await self._run_batched_decode_step_async(
            input_ids_arr,
            attention_mask_arr,
            position_ids_arr,
            logits_to_keep_arr,
            slots,
            kv_lens,
            max_kv_len,
        )
        t1 = time.perf_counter()

        self._total_decode_time += t1 - t0
        self._decode_count += 1
        self._batched_decode_count += 1

        results: List[Tuple[str, int, bool, str, float]] = []
        for j, req in enumerate(active_requests):
            token_start = time.perf_counter()
            req_logits = logits[j, 0, :]

            next_token = _sample_token(req_logits, req.temperature, req.top_p)

            req.generated_ids.append(next_token)
            req.next_token = next_token
            req.kv_len += 1
            self._total_tokens_generated += 1

            finished = False
            finish_reason = None
            if next_token == EOS_TOKEN_ID:
                finished = True
                finish_reason = "stop"
            elif len(req.generated_ids) - req.prompt_len >= req.max_tokens:
                finished = True
                finish_reason = "length"

            token_duration = time.perf_counter() - token_start
            req.token_times.append(token_duration)

            results.append((req.req_id, next_token, finished, finish_reason, token_duration))

        return results

    # ------------------------------------------------------------------ #
    # Request lifecycle
    # ------------------------------------------------------------------ #
    def finish_request(self, req: ReqState, reason: str = "stop") -> None:
        """Mark a request as finished and free its slot."""
        req.finished = True
        req.finished_reason = reason
        if 0 <= req.slot < len(self._slots):
            self._slots[req.slot] = None
            self._free_slots.append(req.slot)
            self._reset_slot_to_zeros(req.slot)

    def get_active_requests(self) -> List[ReqState]:
        """Return all currently active (unfinished) requests."""
        return [req for req in self._slots if req is not None and not req.finished]

    def get_pending_count(self) -> int:
        """Return count of pending requests (not yet assigned a slot)."""
        # This is tracked by the server, not the batcher
        return 0

    def get_metrics(self) -> Dict[str, Any]:
        """Return batcher metrics."""
        avg_prefill = (
            self._total_prefill_time / self._prefill_count
            if self._prefill_count
            else 0.0
        )
        avg_decode = (
            self._total_decode_time / self._decode_count
            if self._decode_count
            else 0.0
        )
        return {
            "max_batch_size": self.max_batch_size,
            "active_slots": len(self.get_active_requests()),
            "free_slots": len(self._free_slots),
            "avg_prefill_ms": round(avg_prefill * 1000, 2),
            "avg_decode_ms": round(avg_decode * 1000, 2),
            "prefill_count": self._prefill_count,
            "decode_count": self._decode_count,
            "fast_path_count": self._fast_path_count,
            "batched_decode_count": self._batched_decode_count,
            "total_tokens_generated": self._total_tokens_generated,
            "tokenizer_cache_size": len(self._tokenizer_cache.cache),
        }
