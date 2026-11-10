"""
CPU vs NPU comparison tool for LFM3-1.7B inference.

Runs the same prompt on both CPU and NPU models and compares:
- Output text similarity
- Generation speed (tokens/sec)
- Prefill latency
- Decode latency
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .config import Config
from .inference import HEAD_DIM, NUM_KV_HEADS


def _load_layer_types(model_path: str) -> Tuple[List[int], List[int]]:
    """Load layer types from config.json alongside the model."""
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        layer_types = cfg.get("layer_types", [])
        conv_layers = [i for i, t in enumerate(layer_types) if t == "conv"]
        attn_layers = [i for i, t in enumerate(layer_types) if "attention" in t]
        return conv_layers, attn_layers
    # Fallback: infer from input names
    return [], []


def _build_initial_state(
    conv_layers: List[int],
    attn_layers: List[int],
    batch_size: int = 1,
    seq_len: int = 0,
    hidden: int = 2048,
    conv_l_cache: int = 7,
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
            (batch_size, NUM_KV_HEADS, seq_len, HEAD_DIM), dtype=np.float32
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, NUM_KV_HEADS, seq_len, HEAD_DIM), dtype=np.float32
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


def _generate_with_timing(
    sess: ort.InferenceSession,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    conv_layers: List[int],
    attn_layers: List[int],
    hidden: int = 2048,
    conv_l_cache: int = 7,
) -> Dict[str, Any]:
    """Generate text with detailed timing.

    Handles both CPU model (past_conv + past_key_values inputs,
    present_conv + present.* outputs) and NPU model
    (past_key_values inputs, past_key_values.* outputs).
    """
    encoded = tokenizer.encode(prompt)
    input_ids_list = encoded.ids
    prompt_len: int = len(input_ids_list)

    input_names = {inp.name for inp in sess.get_inputs()}
    output_names = {o.name for o in sess.get_outputs()}
    needs_past_conv = any("past_conv." in name for name in input_names)
    needs_past_kv = any("past_key_values" in name for name in input_names)

    # Build initial state (empty KV cache, seq_len=0)
    past_conv, past_kv = _build_initial_state(
        conv_layers, attn_layers, seq_len=0, hidden=hidden, conv_l_cache=conv_l_cache
    )

    # Prefill
    input_ids_arr = np.array([input_ids_list], dtype=np.int64)
    attention_mask_arr = np.ones((1, prompt_len), dtype=np.int64)
    position_ids_arr = np.array([list(range(prompt_len))], dtype=np.int64)
    logits_to_keep_arr = np.array([1], dtype=np.int64)

    feed: Dict[str, np.ndarray] = {
        "input_ids": input_ids_arr,
        "attention_mask": attention_mask_arr,
        "position_ids": position_ids_arr,
        "logits_to_keep": logits_to_keep_arr,
    }

    if needs_past_conv:
        feed.update(past_conv)
    if needs_past_kv:
        feed.update(past_kv)

    prefill_start = time.perf_counter()
    outputs = sess.run(None, feed)
    prefill_duration = time.perf_counter() - prefill_start

    logits = outputs[0]
    res = dict(zip([o.name for o in sess.get_outputs()], outputs))

    # Update state from prefill outputs
    past_conv, past_kv = _update_state(
        past_conv, past_kv, res, conv_layers, attn_layers
    )

    next_logits = logits[0, -1, :]
    generated_ids: List[int] = list(input_ids_list)

    # Decode timing
    decode_start = time.perf_counter()
    token_times: List[float] = []

    for step in range(max_new_tokens):
        token_start = time.perf_counter()
        next_token = _sample_token(next_logits, temperature, top_p)
        generated_ids.append(next_token)

        if next_token == 124900:
            break

        seq_len = prompt_len + step + 1
        input_ids_arr = np.array([[next_token]], dtype=np.int64)
        attention_mask_arr = np.ones((1, seq_len), dtype=np.int64)
        position_ids_arr = np.array([[seq_len - 1]], dtype=np.int64)
        logits_to_keep_arr = np.array([1], dtype=np.int64)

        feed = {
            "input_ids": input_ids_arr,
            "attention_mask": attention_mask_arr,
            "position_ids": position_ids_arr,
            "logits_to_keep": logits_to_keep_arr,
        }

        if needs_past_conv:
            feed.update(past_conv)
        if needs_past_kv:
            feed.update(past_kv)

        outputs = sess.run(None, feed)
        logits = outputs[0]
        res = dict(zip([o.name for o in sess.get_outputs()], outputs))

        # Update state from outputs
        past_conv, past_kv = _update_state(
            past_conv, past_kv, res, conv_layers, attn_layers
        )

        next_logits = logits[0, -1, :]

        token_duration = time.perf_counter() - token_start
        token_times.append(token_duration)

    decode_duration = time.perf_counter() - decode_start

    text = tokenizer.decode(generated_ids[prompt_len:])

    return {
        "prompt": prompt,
        "prompt_tokens": prompt_len,
        "generated_tokens": len(generated_ids) - prompt_len,
        "generated_text": text,
        "prefill_duration": prefill_duration,
        "decode_duration": decode_duration,
        "token_times": token_times,
        "avg_token_time": np.mean(token_times) if token_times else 0.0,
        "tokens_per_second": len(generated_ids) - prompt_len
        / decode_duration
        if decode_duration > 0
        else 0.0,
    }


def compare(
    cpu_model_path: str,
    npu_model_path: str,
    tokenizer_path: str,
    prompts: List[str],
    max_tokens: int = 30,
    temperature: float = 0.7,
    top_p: float = 0.9,
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """Compare CPU and NPU inference on the same prompts."""
    custom_ops_lib = (
        config.get("npu.custom_ops_lib")
        if config
        else "/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib/libonnxruntime_vitis_ai_custom_ops.so"
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Get layer types from config
    conv_layers, attn_layers = _load_layer_types(cpu_model_path)
    if not conv_layers:
        # Fallback: hardcoded based on known LFM3 architecture
        conv_layers = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        attn_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    hidden = 2048
    conv_l_cache = 7

    # Load CPU model
    cpu_so = ort.SessionOptions()
    cpu_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu_sess = ort.InferenceSession(
        cpu_model_path, sess_options=cpu_so, providers=["CPUExecutionProvider"]
    )

    # Load NPU model
    npu_so = ort.SessionOptions()
    npu_so.register_custom_ops_library(custom_ops_lib)
    npu_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    npu_sess = ort.InferenceSession(
        npu_model_path, sess_options=npu_so, providers=["VitisAIExecutionProvider"]
    )

    results: Dict[str, Any] = {
        "prompts": [],
        "summary": {},
    }

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 60)

        # CPU generation
        cpu_result = _generate_with_timing(
            cpu_sess, tokenizer, prompt, max_tokens, temperature, top_p,
            conv_layers, attn_layers, hidden, conv_l_cache,
        )
        print(f"  CPU:  {cpu_result['generated_text'][:60]}...")
        print(f"  CPU:  {cpu_result['generated_tokens']} tokens, "
              f"{cpu_result['tokens_per_second']:.1f} tok/s, "
              f"prefill={cpu_result['prefill_duration']*1000:.1f}ms")

        # NPU generation
        npu_result = _generate_with_timing(
            npu_sess, tokenizer, prompt, max_tokens, temperature, top_p,
            conv_layers, attn_layers, hidden, conv_l_cache,
        )
        print(f"  NPU:  {npu_result['generated_text'][:60]}...")
        print(f"  NPU:  {npu_result['generated_tokens']} tokens, "
              f"{npu_result['tokens_per_second']:.1f} tok/s, "
              f"prefill={npu_result['prefill_duration']*1000:.1f}ms")

        # Comparison
        text_match = cpu_result["generated_text"] == npu_result["generated_text"]
        speed_ratio = (
            npu_result["tokens_per_second"] / cpu_result["tokens_per_second"]
            if cpu_result["tokens_per_second"] > 0
            else 0.0
        )
        prefill_ratio = (
            npu_result["prefill_duration"] / cpu_result["prefill_duration"]
            if cpu_result["prefill_duration"] > 0
            else 0.0
        )

        print(f"  Text match: {text_match}")
        print(f"  NPU/CPU speed: {speed_ratio:.2f}x")
        print(f"  NPU/CPU prefill ratio: {prefill_ratio:.2f}x")

        results["prompts"].append({
            "prompt": prompt,
            "cpu": cpu_result,
            "npu": npu_result,
            "text_match": text_match,
            "speed_ratio": speed_ratio,
            "prefill_ratio": prefill_ratio,
        })

    # Summary
    total_prompts = len(results["prompts"])
    matches = sum(1 for p in results["prompts"] if p["text_match"])
    avg_cpu_tps = np.mean([p["cpu"]["tokens_per_second"] for p in results["prompts"]])
    avg_npu_tps = np.mean([p["npu"]["tokens_per_second"] for p in results["prompts"]])
    avg_cpu_prefill = np.mean([p["cpu"]["prefill_duration"] for p in results["prompts"]])
    avg_npu_prefill = np.mean([p["npu"]["prefill_duration"] for p in results["prompts"]])

    results["summary"] = {
        "total_prompts": total_prompts,
        "text_matches": matches,
        "match_rate": matches / total_prompts if total_prompts > 0 else 0.0,
        "avg_cpu_tokens_per_sec": avg_cpu_tps,
        "avg_npu_tokens_per_sec": avg_npu_tps,
        "avg_speed_ratio": avg_npu_tps / avg_cpu_tps if avg_cpu_tps > 0 else 0.0,
        "avg_cpu_prefill_ms": avg_cpu_prefill * 1000,
        "avg_npu_prefill_ms": avg_npu_prefill * 1000,
    }

    return results


def compare_cli(
    cpu_model: str,
    npu_model: str,
    tokenizer: str,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    config: Optional[Config] = None,
) -> int:
    """CLI entry point for comparison."""
    results = compare(
        cpu_model, npu_model, tokenizer, prompts,
        max_tokens, temperature, top_p, config,
    )
    return 0