#!/usr/bin/env python3
"""
Profile CPU ONNX server memory arena fragmentation over time.

Runs N sequential inference requests and measures decode latency per request.
If latency creeps up over time, it indicates memory arena fragmentation.

Usage:
    python3 scripts/profile_cpu_degradation.py --requests 200 --max-tokens 50
"""
import argparse
import sys
import time
import json
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

MODEL_PATH = "/media/scott/data/lfm3_oga_export/model.onnx"
TOKENIZER_PATH = "/media/scott/data/lfm3_oga_export/tokenizer.json"
CONFIG_PATH = "/media/scott/data/lfm3_oga_export/model.onnx.config.json"
# Try common config locations
for cp in [
    "/media/scott/data/lfm3_oga_export/config.json",
    "/media/scott/data/lfm3_oga_export/model.onnx.config.json",
    os.path.join(os.path.dirname(MODEL_PATH), "config.json"),
]:
    if os.path.exists(cp):
        CONFIG_PATH = cp
        break


def load_layer_types(model_path: str):
    model_dir = os.path.dirname(model_path)
    for cp in [
        os.path.join(model_dir, "config.json"),
        model_path + ".config.json",
    ]:
        if os.path.exists(cp):
            with open(cp) as f:
                cfg = json.load(f)
            layer_types = cfg.get("layer_types", [])
            conv_layers = [i for i, t in enumerate(layer_types) if t == "conv"]
            attn_layers = [i for i, t in enumerate(layer_types) if "attention" in t]
            hidden = cfg.get("hidden_size", 2048)
            conv_l_cache = cfg.get("conv_L_cache", 7)
            return conv_layers, attn_layers, hidden, conv_l_cache
    # Fallback defaults
    return [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], 2048, 7


CONV_LAYERS, ATTN_LAYERS, HIDDEN, CONV_L_CACHE = load_layer_types(MODEL_PATH)
NUM_KV_HEADS = 4
HEAD_DIM = 128
USE_KV_QUANT = False


def _build_initial_state(conv_layers, attn_layers, batch_size=1, seq_len=0,
                         hidden=2048, conv_l_cache=7,
                         num_kv_heads=4, head_dim=128, use_kv_quant=False):
    past_conv = {}
    past_kv = {}
    for i in conv_layers:
        past_conv[f"past_conv.{i}"] = np.zeros(
            (batch_size, hidden, conv_l_cache), dtype=np.float32
        )
    for i in attn_layers:
        dtype = np.uint8 if use_kv_quant else np.float32
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype
        )
    return past_conv, past_kv


def _update_state(past_conv, past_kv, res, conv_layers, attn_layers, use_kv_quant=False):
    for i in conv_layers:
        key = f"present_conv.{i}"
        if key in res:
            past_conv[f"past_conv.{i}"] = res[key]
    for i in attn_layers:
        for suffix in ("key", "value"):
            key = f"present.{i}.{suffix}" if not use_kv_quant else f"present.{i}.{suffix}_quantized"
            if key in res:
                past_kv[f"past_key_values.{i}.{suffix}"] = res[key]
    return past_conv, past_kv


def _sample_token(logits, temperature=0.7, top_p=0.9):
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


def generate_once(sess, tokenizer, prompt: str, max_tokens: int):
    encoded = tokenizer.encode(prompt)
    input_ids_list = encoded.ids
    prompt_len = len(input_ids_list)

    past_conv, past_kv = _build_initial_state(
        CONV_LAYERS, ATTN_LAYERS,
        seq_len=0, hidden=HIDDEN, conv_l_cache=CONV_L_CACHE,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        use_kv_quant=USE_KV_QUANT,
    )

    input_ids_arr = np.array([input_ids_list], dtype=np.int64)
    attention_mask_arr = np.ones((1, prompt_len), dtype=np.int64)
    position_ids_arr = np.array([list(range(prompt_len))], dtype=np.int64)
    logits_to_keep_arr = np.array([1], dtype=np.int64)

    feed = {
        "input_ids": input_ids_arr,
        "attention_mask": attention_mask_arr,
        "position_ids": position_ids_arr,
        "logits_to_keep": logits_to_keep_arr,
    }
    feed.update(past_conv)
    feed.update(past_kv)

    outputs = sess.run(None, feed)
    output_names = [o.name for o in sess.get_outputs()]
    res = dict(zip(output_names, outputs))
    logits = res["logits"]
    past_conv, past_kv = _update_state(
        past_conv, past_kv, res, CONV_LAYERS, ATTN_LAYERS,
        use_kv_quant=USE_KV_QUANT,
    )
    next_logits = logits[0, -1, :]
    generated_ids = list(input_ids_list)

    decode_start = time.perf_counter()
    tokens = 0

    for step in range(max_tokens):
        next_token = _sample_token(next_logits, 0.7, 0.9)
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
        feed.update(past_conv)
        feed.update(past_kv)

        outputs = sess.run(None, feed)
        logits = outputs[0]
        next_logits = logits[0, -1, :]

        res = dict(zip(output_names, outputs))
        past_conv, past_kv = _update_state(
            past_conv, past_kv, res, CONV_LAYERS, ATTN_LAYERS,
        )

        tokens += 1

    decode_duration = time.perf_counter() - decode_start
    return decode_duration, tokens


def main():
    parser = argparse.ArgumentParser(description="Profile CPU ONNX degradation")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--max-tokens", type=int, default=30, help="Max tokens per request")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt")
    args = parser.parse_args()

    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 4
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = False  # Disabled to prevent fragmentation
    so.enable_mem_reuse = False
    so.enable_mem_pattern = False
    so.add_session_config_entry("optimize_model", "1")
    so.add_session_config_entry("enable_bf16", "1")

    sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    print(f"Model loaded. Conv layers: {CONV_LAYERS}, Attention layers: {ATTN_LAYERS}")
    print(f"Running {args.requests} requests with {args.max_tokens} max tokens...")

    results = []

    for i in range(args.requests):
        duration, tokens = generate_once(sess, tokenizer, args.prompt, args.max_tokens)
        tps = tokens / duration if duration > 0 else 0
        results.append({
            "request": i + 1,
            "duration": duration,
            "tokens": tokens,
            "tps": tps,
        })

        if (i + 1) % 25 == 0 or (i + 1) == args.requests:
            recent = results[-25:] if len(results) >= 25 else results
            avg_tps = sum(r["tps"] for r in recent) / len(recent)
            avg_dur = sum(r["duration"] for r in recent) / len(recent)
            print(f"  Request {i + 1:3d}: avg TPS={avg_tps:6.1f}, avg dur={avg_dur:.3f}s")

    first_25 = results[:25]
    last_25 = results[-25:]
    first_avg_tps = sum(r["tps"] for r in first_25) / len(first_25)
    last_avg_tps = sum(r["tps"] for r in last_25) / len(last_25)
    first_avg_dur = sum(r["duration"] for r in first_25) / len(first_25)
    last_avg_dur = sum(r["duration"] for r in last_25) / len(last_25)

    print("\n--- Degradation Summary ---")
    print(f"First 25 requests:  avg TPS={first_avg_tps:6.1f}, avg dur={first_avg_dur:.3f}s")
    print(f"Last 25 requests:   avg TPS={last_avg_tps:6.1f}, avg dur={last_avg_dur:.3f}s")
    tps_change = (last_avg_tps - first_avg_tps) / first_avg_tps * 100
    dur_change = (last_avg_dur - first_avg_dur) / first_avg_dur * 100
    print(f"TPS change:         {tps_change:+.1f}%")
    print(f"Duration change:    {dur_change:+.1f}%")

    if dur_change > 10:
        print("\n⚠️  DEGRADATION DETECTED: Decode latency increased >10% over session")
    elif dur_change > 5:
        print("\n⚠️  MILD DEGRADATION: Decode latency increased >5% over session")
    else:
        print("\n✅ No significant degradation detected")


if __name__ == "__main__":
    main()
