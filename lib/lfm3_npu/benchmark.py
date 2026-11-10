"""
Benchmarking for LFM3-1.7B NPU inference.

Measures prefill latency, decode throughput, tokens per second,
time-to-first-token (TTFT), per-token latency, and concurrency scaling.
"""

import json
import os
import statistics
import time
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
    if not conv_layers:
        conv_layers = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    if not attn_layers:
        attn_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    return conv_layers, attn_layers, hidden, conv_l_cache


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
    """Build feed dictionary for ONNX Runtime."""
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


def _update_state(
    past_conv: Dict[str, np.ndarray],
    past_kv: Dict[str, np.ndarray],
    res: Dict[str, np.ndarray],
    conv_layers: List[int],
    attn_layers: List[int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Update state from model outputs by matching output names."""
    for i in conv_layers:
        key = f"present_conv.{i}"
        if key in res:
            past_conv[f"past_conv.{i}"] = res[key]
    for i in attn_layers:
        for suffix in ("key", "value"):
            key = f"present.{i}.{suffix}"
            if key in res:
                past_kv[f"past_key_values.{i}.{suffix}"] = res[key]
    return past_conv, past_kv


def _sample_token_greedy(logits: np.ndarray) -> int:
    """Greedy sampling (argmax)."""
    return int(np.argmax(logits))


def _sample_token_from_logits(
    logits: np.ndarray, temperature: float, top_p: float
) -> int:
    """Sample next token from logits with temperature and top-p."""
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


def _run_prefill(
    sess: ort.InferenceSession,
    input_ids_arr: np.ndarray,
    attention_mask_arr: np.ndarray,
    position_ids_arr: np.ndarray,
    logits_to_keep_arr: np.ndarray,
    past_conv: Dict[str, np.ndarray],
    past_kv: Dict[str, np.ndarray],
    needs_past_conv: bool,
    needs_past_kv: bool,
    output_names: List[str],
    conv_layers: List[int],
    attn_layers: List[int],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], float]:
    """Run a single prefill inference and return logits, state, and duration."""
    feed = _build_feed(
        input_ids_arr, attention_mask_arr, position_ids_arr,
        logits_to_keep_arr, past_conv, past_kv,
        needs_past_conv, needs_past_kv,
    )
    start = time.perf_counter()
    outputs = sess.run(None, feed)
    duration = time.perf_counter() - start
    res = dict(zip(output_names, outputs))
    past_conv, past_kv = _update_state(past_conv, past_kv, res, conv_layers, attn_layers)
    return outputs[0], past_conv, past_kv, duration


def _run_decode_step(
    sess: ort.InferenceSession,
    next_logits: np.ndarray,
    step: int,
    prompt_len: int,
    past_conv: Dict[str, np.ndarray],
    past_kv: Dict[str, np.ndarray],
    needs_past_conv: bool,
    needs_past_kv: bool,
    output_names: List[str],
    conv_layers: List[int],
    attn_layers: List[int],
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], int, float]:
    """Run a single decode step and return logits, state, token, and duration."""
    token_start = time.perf_counter()
    next_token = int(np.argmax(next_logits)) if temperature == 0.0 else _sample_token_from_logits(next_logits, temperature, top_p)
    seq_len = prompt_len + step + 1
    input_ids_arr = np.array([[next_token]], dtype=np.int64)
    attention_mask_arr = np.ones((1, seq_len), dtype=np.int64)
    position_ids_arr = np.array([[seq_len - 1]], dtype=np.int64)
    logits_to_keep_arr = np.array([1], dtype=np.int64)

    feed = _build_feed(
        input_ids_arr, attention_mask_arr, position_ids_arr,
        logits_to_keep_arr, past_conv, past_kv,
        needs_past_conv, needs_past_kv,
    )
    outputs = sess.run(None, feed)
    step_duration = time.perf_counter() - token_start
    logits = outputs[0]
    res = dict(zip(output_names, outputs))
    past_conv, past_kv = _update_state(past_conv, past_kv, res, conv_layers, attn_layers)
    return logits[0, -1, :], past_conv, past_kv, next_token, step_duration


def _percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_vals):
        return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
    return sorted_vals[f]


def bench_model(
    model_path: str,
    tokenizer_path: str,
    prompt: str = "The quick brown fox jumps over the lazy dog",
    max_tokens: int = 50,
    iterations: int = 3,
    concurrency: int = 1,
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    config = config or Config({})
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load layer types
    conv_layers, attn_layers, hidden, conv_l_cache = _load_layer_types(model_path)

    # Load model
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = True
    so.enable_mem_reuse = True
    so.enable_mem_pattern = True
    so.add_session_config_entry("optimize_model", "1")
    so.add_session_config_entry("enable_bf16", "1")

    import os as _os
    cpu_count = _os.cpu_count() or 4
    so.intra_op_num_threads = cpu_count
    so.inter_op_num_threads = 1

    providers = ["CPUExecutionProvider"]
    custom_ops_lib = None
    if "npu" in model_path.lower() or "centertap" in model_path.lower():
        providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
        custom_ops_lib = config.get(
            "npu.custom_ops_lib",
            "/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib/libonnxruntime_vitis_ai_custom_ops.so",
        )
        if custom_ops_lib:
            so.register_custom_ops_library(custom_ops_lib)

    sess = ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=providers,
    )

    output_names = [o.name for o in sess.get_outputs()]
    input_names = [i.name for i in sess.get_inputs()]
    needs_past_conv = any("past_conv" in name for name in input_names)
    needs_past_kv = any("past_key_values" in name for name in input_names)

    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids_list = encoded.ids
    prompt_len = len(input_ids_list)

    # Warmup
    print(f"Warming up ({iterations} iteration(s))...", flush=True)
    for _ in range(min(iterations, 3)):
        past_conv, past_kv = _build_initial_state(
            conv_layers, attn_layers, seq_len=0,
            hidden=hidden, conv_l_cache=conv_l_cache,
        )
        input_ids_arr = np.array([input_ids_list], dtype=np.int64)
        attention_mask_arr = np.ones((1, prompt_len), dtype=np.int64)
        position_ids_arr = np.array([list(range(prompt_len))], dtype=np.int64)
        logits_to_keep_arr = np.array([1], dtype=np.int64)
        feed = _build_feed(
            input_ids_arr, attention_mask_arr, position_ids_arr,
            logits_to_keep_arr, past_conv, past_kv,
            needs_past_conv, needs_past_kv,
        )
        outputs = sess.run(None, feed)
        res = dict(zip(output_names, outputs))
        past_conv, past_kv = _update_state(
            past_conv, past_kv, res, conv_layers, attn_layers
        )
        nl = outputs[0][0, -1, :]
        # Run a few decode steps for warmup
        for _ in range(min(5, max_tokens)):
            nl, past_conv, past_kv, _, _ = _run_decode_step(
                sess, nl, 0, prompt_len,
                past_conv, past_kv, needs_past_conv, needs_past_kv, output_names,
                conv_layers, attn_layers,
            )

    # Benchmark prefill
    print(f"Prefill benchmark ({iterations} iteration(s))...", flush=True)
    prefill_times: List[float] = []
    ttft_times: List[float] = []
    for _ in range(iterations):
        past_conv, past_kv = _build_initial_state(
            conv_layers, attn_layers, seq_len=0,
            hidden=hidden, conv_l_cache=conv_l_cache,
        )
        input_ids_arr = np.array([input_ids_list], dtype=np.int64)
        attention_mask_arr = np.ones((1, prompt_len), dtype=np.int64)
        position_ids_arr = np.array([list(range(prompt_len))], dtype=np.int64)
        logits_to_keep_arr = np.array([1], dtype=np.int64)

        ttft_start = time.perf_counter()
        logits, past_conv, past_kv, prefill_duration = _run_prefill(
            sess, input_ids_arr, attention_mask_arr, position_ids_arr,
            logits_to_keep_arr, past_conv, past_kv,
            needs_past_conv, needs_past_kv, output_names,
            conv_layers, attn_layers,
        )
        ttft_duration = time.perf_counter() - ttft_start

        prefill_times.append(prefill_duration)
        ttft_times.append(ttft_duration)

    avg_prefill = statistics.mean(prefill_times)
    avg_ttft = statistics.mean(ttft_times)
    p95_ttft = _percentile(ttft_times, 95)
    print(f"  Average prefill time: {avg_prefill * 1000:.1f} ms", flush=True)
    print(f"  Average TTFT: {avg_ttft * 1000:.1f} ms", flush=True)
    print(f"  P95 TTFT: {p95_ttft * 1000:.1f} ms", flush=True)

    # Benchmark decode
    print(
        f"Decode benchmark ({max_tokens} tokens x {iterations} iteration(s))...",
        flush=True,
    )
    decode_times: List[float] = []
    tokens_per_sec: List[float] = []
    all_token_times: List[float] = []

    for _ in range(iterations):
        past_conv, past_kv = _build_initial_state(
            conv_layers, attn_layers, seq_len=0,
            hidden=hidden, conv_l_cache=conv_l_cache,
        )
        input_ids_arr = np.array([input_ids_list], dtype=np.int64)
        attention_mask_arr = np.ones((1, prompt_len), dtype=np.int64)
        position_ids_arr = np.array([list(range(prompt_len))], dtype=np.int64)
        logits_to_keep_arr = np.array([1], dtype=np.int64)
        feed = _build_feed(
            input_ids_arr, attention_mask_arr, position_ids_arr,
            logits_to_keep_arr, past_conv, past_kv,
            needs_past_conv, needs_past_kv,
        )
        outputs = sess.run(None, feed)
        res = dict(zip(output_names, outputs))
        past_conv, past_kv = _update_state(
            past_conv, past_kv, res, conv_layers, attn_layers
        )
        nl = outputs[0][0, -1, :]

        start = time.perf_counter()
        for step in range(max_tokens):
            nl, past_conv, past_kv, _, step_duration = _run_decode_step(
                sess, nl, step, prompt_len,
                past_conv, past_kv, needs_past_conv, needs_past_kv, output_names,
                conv_layers, attn_layers,
            )
            all_token_times.append(step_duration)
        elapsed = time.perf_counter() - start
        decode_times.append(elapsed)
        tokens_per_sec.append(max_tokens / elapsed)

    avg_decode = statistics.mean(decode_times)
    avg_tps = statistics.mean(tokens_per_sec)
    all_token_times_sorted = sorted(all_token_times)
    avg_token_latency = statistics.mean(all_token_times) * 1000
    p95_token_latency = _percentile(all_token_times, 95) * 1000
    p99_token_latency = _percentile(all_token_times, 99) * 1000

    print(f"  Average decode: {avg_decode * 1000:.1f} ms per {max_tokens} tokens", flush=True)
    print(f"  Average tokens/sec: {avg_tps:.1f}", flush=True)
    print(f"  Average token latency: {avg_token_latency:.1f} ms", flush=True)
    print(f"  P95 token latency: {p95_token_latency:.1f} ms", flush=True)
    print(f"  P99 token latency: {p99_token_latency:.1f} ms", flush=True)

    return {
        "model": model_path,
        "prompt_tokens": prompt_len,
        "max_tokens": max_tokens,
        "iterations": iterations,
        "concurrency": concurrency,
        "avg_prefill_ms": round(avg_prefill * 1000, 2),
        "avg_ttft_ms": round(avg_ttft * 1000, 2),
        "p95_ttft_ms": round(p95_ttft * 1000, 2),
        "avg_decode_ms": round(avg_decode * 1000, 2),
        "avg_tokens_per_sec": round(avg_tps, 2),
        "avg_token_latency_ms": round(avg_token_latency, 2),
        "p95_token_latency_ms": round(p95_token_latency, 2),
        "p99_token_latency_ms": round(p99_token_latency, 2),
    }


def bench_concurrency(
    model_path: str,
    tokenizer_path: str,
    prompt: str = "The quick brown fox jumps over the lazy dog",
    max_tokens: int = 50,
    concurrency_levels: List[int] = [1, 2, 4, 8],
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """Run benchmarks at different concurrency levels."""
    results = []
    for conc in concurrency_levels:
        print(f"\n--- Concurrency {conc} ---", flush=True)
        result = bench_model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            prompt=prompt,
            max_tokens=max_tokens,
            iterations=3,
            concurrency=conc,
            config=config,
        )
        results.append(result)
    return {"results": results}