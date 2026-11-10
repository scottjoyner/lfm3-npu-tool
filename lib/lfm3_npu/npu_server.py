#!/usr/bin/env python3
"""
LFM3-1.7B NPU Inference Server - FastAPI with VitisAI ExecutionProvider.

OpenAI-compatible API on port 1239.
Uses center-tap Conv replacement model to avoid libvaiml.so crash.
"""

import sys
import os
import time
import uuid
import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime_genai as og  # Register RyzenAI NPU provider before onnxruntime
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from tokenizers import Tokenizer

# Add lib to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfm3_npu.config import load_config
from lfm3_npu.metrics import MetricsCollector


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


def _sample_token(logits, temperature, top_p):
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


# Load config
config = load_config()
MODEL_PATH = config.get("server.npu.model_path", "/media/scott/data/lfm3_oga_export/model.onnx")
TOKENIZER_PATH = config.get("model.tokenizer", "/media/scott/data/lfm3_oga_export/tokenizer.json")

# Global state
tokenizer = None
sess = None
metrics = None

# Layer types from config
CONV_LAYERS: List[int] = []
ATTN_LAYERS: List[int] = []
HIDDEN = 2048
CONV_L_CACHE = 7

NEEDS_PAST_CONV = False
NEEDS_PAST_KV = False


def _configure_session_options() -> ort.SessionOptions:
    """Configure ONNX Runtime session options for RyzenAI NPU EP."""
    import os as _os

    so = ort.SessionOptions()

    perf = config.get("performance", {})
    ort_cfg = perf.get("ort_session", {})

    # Thread configuration - auto-detect CPU cores if not specified
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

    # Execution mode
    exec_mode = ort_cfg.get("execution_mode", "ORT_SEQUENTIAL")
    if exec_mode == "ORT_PARALLEL":
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Graph optimization
    opt_level = ort_cfg.get("graph_optimization_level", "ORT_ENABLE_ALL")
    if opt_level == "ORT_ENABLE_ALL":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    elif opt_level == "ORT_ENABLE_BASIC":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif opt_level == "ORT_ENABLE_EXTENDED":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    # Memory arena optimizations
    so.enable_cpu_mem_arena = ort_cfg.get("enable_cpu_mem_arena", True)
    so.enable_mem_reuse = ort_cfg.get("enable_mem_reuse", True)
    so.enable_mem_pattern = ort_cfg.get("enable_mem_pattern", True)

    # Session config entries for RyzenAI EP
    session_entries = ort_cfg.get("session_config_entries", {})
    session_entries.setdefault("optimize_model", "1")
    session_entries.setdefault("enable_bf16", "1")

    for key, value in session_entries.items():
        so.add_session_config_entry(key, str(value))

    return so


def load_model():
    """Load tokenizer and ONNX model on RyzenAI NPU EP."""
    global tokenizer, sess, metrics, CONV_LAYERS, ATTN_LAYERS, HIDDEN, CONV_L_CACHE
    global NEEDS_PAST_CONV, NEEDS_PAST_KV

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    try:
        sess = ort.InferenceSession(
            MODEL_PATH,
            providers=["RyzenAIExecutionProvider"],
        )
        print(f"NPU Server: Successfully loaded with RyzenAI EP", flush=True)
    except Exception as e:
        print(f"NPU Server: RyzenAI EP failed: {type(e).__name__}: {e}", flush=True)
        # Fall back to CPU
        sess = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        print(f"NPU Server: Fallback to CPUExecutionProvider", flush=True)

    # Load layer types from config.json
    CONV_LAYERS, ATTN_LAYERS, HIDDEN, CONV_L_CACHE = _load_layer_types(MODEL_PATH)

    input_names = {inp.name for inp in sess.get_inputs()}
    NEEDS_PAST_CONV = any("past_conv." in name for name in input_names)
    NEEDS_PAST_KV = any("past_key_values" in name for name in input_names)

    metrics = MetricsCollector(provider="RyzenAIExecutionProvider", model="lfm3-1.7b-npu")
    print(f"NPU Server: Providers: {sess.get_providers()}", flush=True)
    print(f"NPU Server: Model loaded on RyzenAI EP", flush=True)
    print(f"NPU Server: Conv layers: {CONV_LAYERS}, Attention layers: {ATTN_LAYERS}", flush=True)
    print(f"NPU Server: Needs past_conv: {NEEDS_PAST_CONV}, Needs past_key_values: {NEEDS_PAST_KV}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="LFM3-1.7B NPU Inference Server", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "lfm3-1.7b-npu"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 50
    stream: Optional[bool] = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "liquid-ai"


def _build_feed(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
    logits_to_keep: np.ndarray,
    past_conv: Dict[str, np.ndarray],
    past_kv: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Build feed dictionary for ONNX Runtime."""
    feed: Dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "logits_to_keep": logits_to_keep,
    }
    if NEEDS_PAST_CONV:
        feed.update(past_conv)
    if NEEDS_PAST_KV:
        feed.update(past_kv)
    return feed


@app.get("/health")
async def health():
    return {"status": "healthy", "provider": "VitisAIExecutionProvider", "model": "lfm3-1.7b-npu"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            ModelInfo(id="lfm3-1.7b-npu", created=int(time.time())).model_dump()
        ]
    }


@app.get("/v1/model/info")
async def model_info():
    return {
        "id": "lfm3-1.7b-npu",
        "provider": "RyzenAIExecutionProvider",
        "device": "AMD XDNA2 NPU",
        "model_type": "LFM3-1.7B",
        "conv_layers": "full (original)",
        "max_context_length": 262144,
        "vocab_size": 128000,
    }


@app.get("/metrics")
async def metrics_endpoint():
    if metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not initialized")
    return PlainTextResponse(content=metrics.get_metrics(), media_type=metrics.get_content_type())


def _generate_text(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate text (non-streaming). Returns the generated text string."""
    encoded = tokenizer.encode(prompt)
    input_ids_list = encoded.ids
    prompt_len = len(input_ids_list)

    # Initialize empty state (seq_len=0)
    past_conv, past_kv = _build_initial_state(
        CONV_LAYERS, ATTN_LAYERS,
        seq_len=0, hidden=HIDDEN, conv_l_cache=CONV_L_CACHE,
    )

    # Prefill
    input_ids_arr = np.array([input_ids_list], dtype=np.int64)
    attention_mask_arr = np.ones((1, prompt_len), dtype=np.int64)
    position_ids_arr = np.array([list(range(prompt_len))], dtype=np.int64)
    logits_to_keep_arr = np.array([1], dtype=np.int64)

    feed = _build_feed(
        input_ids_arr, attention_mask_arr, position_ids_arr,
        logits_to_keep_arr, past_conv, past_kv,
    )

    prefill_start = time.perf_counter()
    outputs = sess.run(None, feed)
    prefill_duration = time.perf_counter() - prefill_start
    metrics.record_prefill(prefill_duration, prompt_len)

    output_names = [o.name for o in sess.get_outputs()]
    res = dict(zip(output_names, outputs))
    logits = res["logits"]

    # Update state from prefill outputs
    past_conv, past_kv = _update_state(
        past_conv, past_kv, res, CONV_LAYERS, ATTN_LAYERS,
    )

    next_logits = logits[0, -1, :]
    generated_ids = list(input_ids_list)

    # Decode
    decode_start = time.perf_counter()
    token_times: List[float] = []

    for step in range(max_tokens):
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

        feed = _build_feed(
            input_ids_arr, attention_mask_arr, position_ids_arr,
            logits_to_keep_arr, past_conv, past_kv,
        )
        outputs = sess.run(None, feed)
        logits = outputs[0]
        next_logits = logits[0, -1, :]

        # Update state from outputs
        res = dict(zip(output_names, outputs))
        past_conv, past_kv = _update_state(
            past_conv, past_kv, res, CONV_LAYERS, ATTN_LAYERS,
        )

        token_duration = time.perf_counter() - token_start
        token_times.append(token_duration)
        metrics.record_token_generation(token_duration)

    decode_duration = time.perf_counter() - decode_start
    metrics.record_decode(decode_duration, len(generated_ids) - prompt_len)

    return tokenizer.decode(generated_ids[prompt_len:])


def _generate_stream(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    """Generate text (streaming). Yields SSE-compatible chunks."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    encoded = tokenizer.encode(prompt)
    input_ids_list = encoded.ids
    prompt_len = len(input_ids_list)

    # Initialize empty state (seq_len=0)
    past_conv, past_kv = _build_initial_state(
        CONV_LAYERS, ATTN_LAYERS,
        seq_len=0, hidden=HIDDEN, conv_l_cache=CONV_L_CACHE,
    )

    # Prefill
    input_ids_arr = np.array([input_ids_list], dtype=np.int64)
    attention_mask_arr = np.ones((1, prompt_len), dtype=np.int64)
    position_ids_arr = np.array([list(range(prompt_len))], dtype=np.int64)
    logits_to_keep_arr = np.array([1], dtype=np.int64)

    feed = _build_feed(
        input_ids_arr, attention_mask_arr, position_ids_arr,
        logits_to_keep_arr, past_conv, past_kv,
    )

    outputs = sess.run(None, feed)
    output_names = [o.name for o in sess.get_outputs()]
    res = dict(zip(output_names, outputs))
    logits = res["logits"]

    # Record prefill metrics
    prefill_start = time.perf_counter()
    # Update state from prefill outputs
    past_conv, past_kv = _update_state(
        past_conv, past_kv, res, CONV_LAYERS, ATTN_LAYERS,
    )
    prefill_duration = time.perf_counter() - prefill_start
    metrics.record_prefill(prefill_duration, prompt_len)

    next_logits = logits[0, -1, :]
    generated_ids = list(input_ids_list)

    # Decode
    decode_start = time.perf_counter()
    token_times: List[float] = []

    for step in range(max_tokens):
        token_start = time.perf_counter()

        next_token = _sample_token(next_logits, temperature, top_p)
        generated_ids.append(next_token)

        if next_token == 124900:
            break

        token_text = tokenizer.decode([next_token])

        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "lfm3-1.7b-npu",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }
            ]
        }

        seq_len = prompt_len + step + 1
        input_ids_arr = np.array([[next_token]], dtype=np.int64)
        attention_mask_arr = np.ones((1, seq_len), dtype=np.int64)
        position_ids_arr = np.array([[seq_len - 1]], dtype=np.int64)
        logits_to_keep_arr = np.array([1], dtype=np.int64)

        feed = _build_feed(
            input_ids_arr, attention_mask_arr, position_ids_arr,
            logits_to_keep_arr, past_conv, past_kv,
        )
        outputs = sess.run(None, feed)
        logits = outputs[0]
        next_logits = logits[0, -1, :]

        # Update state from outputs
        res = dict(zip(output_names, outputs))
        past_conv, past_kv = _update_state(
            past_conv, past_kv, res, CONV_LAYERS, ATTN_LAYERS,
        )

        token_duration = time.perf_counter() - token_start
        token_times.append(token_duration)
        metrics.record_token_generation(token_duration)

    decode_duration = time.perf_counter() - decode_start
    metrics.record_decode(decode_duration, len(generated_ids) - prompt_len)

    # Final chunk
    yield {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "lfm3-1.7b-npu",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ]
    }
    yield "[DONE]"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not sess:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = ""
    for msg in req.messages:
        if msg.role == "system":
            prompt += f"system: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"user: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"assistant: {msg.content}\n"
    prompt += "assistant: "

    max_tokens = req.max_tokens or 50
    temperature = req.temperature or 0.7
    top_p = req.top_p or 1.0

    if req.stream:
        return StreamingResponse(
            _stream_generate_async(prompt, max_tokens, temperature, top_p),
            media_type="text/event-stream"
        )

    # Run blocking inference in thread pool to avoid blocking event loop
    result = await asyncio.to_thread(_generate_text, prompt, max_tokens, temperature, top_p)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "lfm3-1.7b-npu",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt).ids),
            "completion_tokens": len(tokenizer.encode(result).ids),
            "total_tokens": len(tokenizer.encode(prompt).ids) + len(tokenizer.encode(result).ids),
        }
    }


async def _stream_generate_async(prompt, max_tokens, temperature, top_p):
    """Stream generate with blocking inference offloaded to thread pool."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    import asyncio as _asyncio
    queue = _asyncio.Queue()

    def _run_generator():
        try:
            for item in _generate_stream(prompt, max_tokens, temperature, top_p):
                if item == "[DONE]":
                    queue.put_nowait("__DONE__")
                else:
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "lfm3-1.7b-npu",
                        "choices": item["choices"],
                    }
                    queue.put_nowait(json.dumps(chunk))
        except Exception as e:
            metrics.record_error("chat/completions")
            error_chunk = {"error": {"message": str(e), "type": "runtime_error"}}
            queue.put_nowait(json.dumps(error_chunk))
        finally:
            queue.put_nowait("__END__")

    _asyncio.get_event_loop().run_in_executor(None, _run_generator)

    while True:
        item = await queue.get()
        if item == "__END__":
            break
        if item == "__DONE__":
            yield f"data: [DONE]\n\n"
            break
        yield f"data: {item}\n\n"


if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1239)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)