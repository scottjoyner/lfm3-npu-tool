"""
LFM3-1.7B Batched Inference Server - FastAPI with Continuous Batching.

OpenAI-compatible API. Uses ContinuousBatcher for batched prefill + batched decode,
dramatically improving throughput under concurrency.

Self-contained server that imports ContinuousBatcher from batch_inference.py.
"""

import asyncio
import json
import os
import signal
import ssl
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from fastapi import Depends
from pydantic import BaseModel
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfm3_npu.config import load_config
from lfm3_npu.metrics import MetricsCollector
from lfm3_npu.tracing import init_tracer, trace_prefill, trace_decode, trace_request, get_tracer
from lfm3_npu.batch_inference import (
    ContinuousBatcher,
    ReqState,
    _load_layer_types,
    _sample_token,
    EOS_TOKEN_ID,
)
from lfm3_npu.enterprise import RateLimiter, CircuitBreaker, CircuitOpenError, log_event, register_circuit_breaker, circuit_breaker_prometheus_metrics
from lfm3_npu.tls import (tls_ready, ssl_context, https_url, tls_status,
                          generate_dev_certificates, TLS_CERT_FILE, TLS_KEY_FILE,
                          TLS_CA_FILE, TLS_VERIFY_CLIENT)
from lfm3_npu.slo import get_tracker, record_request, prometheus_metrics_all
from lfm3_npu.validation import (
    validate_client, validate_messages, validate_max_tokens,
    ValidationError, validation_error_response, is_validation_error,
)
from lfm3_npu.security_headers import SecurityHeadersMiddleware
from lfm3_npu.audit import get_audit
from lfm3_npu.rbac import RBACAuth, role_required


# ====================================================================== #
# Configuration (from environment or config file)
# ====================================================================== #
config = load_config()
TOKENIZER_PATH = config.get("model.tokenizer", "/media/scott/data/lfm3_oga_export/tokenizer.json")
MAX_BATCH_SIZE = config.get("performance.server.max_concurrent_requests", 8)

MODEL_PATH = os.environ.get(
    "LFM3_MODEL_PATH",
    config.get("server.cpu.model_path", "/media/scott/data/lfm3_oga_export/model.onnx"),
)
PROVIDER = os.environ.get("LFM3_PROVIDER", "CPUExecutionProvider")
CUSTOM_OPS_PATH = os.environ.get(
    "LFM3_CUSTOM_OPS_PATH",
    config.get(
        "npu.custom_ops_lib",
        "/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib/libonnxruntime_vitis_ai_custom_ops.so",
    ),
)
SERVER_NAME = os.environ.get("LFM3_SERVER_NAME", "lfm3-1.7b")
PROVIDER_LABEL = os.environ.get("LFM3_PROVIDER_LABEL", "CPU")
WARMUP_ON_START = os.environ.get("LFM3_WARMUP_ON_START", "false").lower() == "true"
WARMUP_PROMPT = os.environ.get("LFM3_WARMUP_PROMPT", "The quick brown fox")
WARMUP_TOKENS = int(os.environ.get("LFM3_WARMUP_TOKENS", "5"))

# Server port (for fleet status reporting)
SERVER_PORT = int(os.environ.get("LFM3_PORT", "1238"))

# API key for authentication (optional - if set, all endpoints require X-API-Key header)
API_KEY = os.environ.get("LFM3_API_KEY", "")

# Enterprise: rate limiting + circuit breaker
_rate_limiter = RateLimiter()
_circuit_breaker = CircuitBreaker(
    failure_threshold=config.get("circuit_breaker.failure_threshold", 5),
    recovery_timeout=config.get("circuit_breaker.recovery_timeout_seconds", 30.0),
)

print(f"{SERVER_NAME}: Rate limiting {_rate_limiter.rate:.2f} req/s (burst {_rate_limiter.capacity})", flush=True)
print(f"{SERVER_NAME}: Circuit breaker threshold={_circuit_breaker.failure_threshold}, recovery={_circuit_breaker.recovery_timeout}s", flush=True)

# Enterprise: RBAC authentication
_rbac_auth = RBACAuth()
if _rbac_auth.enabled:
    key_infos = _rbac_auth.list_keys()
    roles = {ki["role"] for ki in key_infos.values()}
    print(f"{SERVER_NAME}: RBAC authentication enabled ({len(key_infos)} key(s), roles: {sorted(roles)})", flush=True)
else:
    print(f"{SERVER_NAME}: RBAC authentication disabled (open access)", flush=True)

# Enterprise: SLO tracker
_slo_target_availability = float(os.environ.get("LFM3_SLO_AVAILABILITY", "99.9"))
_slo_target_latency_ms = float(os.environ.get("LFM3_SLO_LATENCY_P95_MS", "5000"))
_slo_tracker = get_tracker(PROVIDER, f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}")

# Enterprise: audit logger
_audit = get_audit()

# Enterprise: register circuit breaker for Prometheus export
register_circuit_breaker(_circuit_breaker, key=f"{PROVIDER}:{SERVER_NAME}-{PROVIDER_LABEL.lower()}")

# Enterprise: load shedding - reject new requests when active slots exceed threshold
_load_shed_max_active = int(os.environ.get("LFM3_LOAD_SHED_MAX_ACTIVE", "100"))

# Enterprise: graceful shutdown state
_shutting_down = False
_shutdown_event_loop = None

# Structured JSON logging (set LFM3_LOG_FORMAT=json)
LOG_FORMAT_JSON = os.environ.get("LFM3_LOG_FORMAT", "").lower() == "json"


def _log_structured(level: str, message: str, **kwargs) -> None:
    """Emit a structured JSON log line."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "server": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
        "provider": PROVIDER,
        "port": SERVER_PORT,
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry), flush=True)


def _log(level: str, message: str, **kwargs) -> None:
    """Log a message - JSON structured if LFM3_LOG_FORMAT=json, else plain text."""
    if LOG_FORMAT_JSON:
        _log_structured(level, message, **kwargs)
    else:
        print(f"[{level}] {message}", flush=True)


# Global state
tokenizer: Optional[Tokenizer] = None
batcher: Optional[ContinuousBatcher] = None
metrics: Optional[MetricsCollector] = None

# Layer types (loaded at startup)
CONV_LAYERS: List[int] = []
ATTN_LAYERS: List[int] = []
HIDDEN = 2048
CONV_L_CACHE = 7

# Server start time (set on first load_model call)
START_TIME: Optional[float] = None

# Batch loop state (module-level for status endpoint access)
_request_queue: Optional[asyncio.Queue] = None
_batch_task: Optional[asyncio.Task] = None
_shutdown_event: Optional[asyncio.Event] = None
_pending: List[ReqState] = []
_active: List[ReqState] = []

# Per-request metrics tracking
_request_metrics: Dict[str, Dict] = {}

# Per-endpoint metrics tracking
_endpoint_metrics: Dict[str, Dict[str, Any]] = {}

# Token usage tracking per API key (hash -> {prompt_tokens, completion_tokens, total_tokens, request_count})
_token_usage: Dict[str, Dict[str, int]] = {}
_token_usage_lock = asyncio.Lock()

# Per-key token quotas (hash -> max_tokens per period, 0 = unlimited)
_token_quotas: Dict[str, int] = {}

# Request tracing - correlation ID support
_REQUEST_ID_HEADER = "X-Request-ID"
_TRACE_ID_HEADER = "X-Trace-ID"

# Trace history - stores last N completed requests for fleet querying
_TRACE_HISTORY_MAX = 100
_trace_history: List[Dict[str, Any]] = []
_trace_history_lock = asyncio.Lock()


async def _record_trace(req_id: str, prompt: str, prompt_tokens: int,
                        completion_tokens: int, total_tokens: int,
                        ttft_ms: float, total_latency_ms: float,
                        finish_reason: str, provider: str,
                        temperature: float, top_p: float,
                        error: bool = False, error_type: str = None) -> None:
    """Record a completed request trace for fleet querying."""
    trace_entry = {
        "req_id": req_id,
        "timestamp": time.time(),
        "provider": provider,
        "prompt": prompt[:100],  # Truncate for storage
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "finish_reason": finish_reason,
        "temperature": temperature,
        "top_p": top_p,
        "error": error,
        "error_type": error_type,
    }
    async with _trace_history_lock:
        _trace_history.append(trace_entry)
        # Keep only the last MAX entries
        if len(_trace_history) > _TRACE_HISTORY_MAX:
            _trace_history.pop(0)


async def _get_trace_history(limit: int = 50, provider: str = None,
                             error_only: bool = False) -> List[Dict[str, Any]]:
    """Get trace history with optional filtering."""
    async with _trace_history_lock:
        traces = list(_trace_history)
    if provider:
        traces = [t for t in traces if t["provider"] == provider]
    if error_only:
        traces = [t for t in traces if t.get("error")]
    # Return most recent first
    traces.reverse()
    return traces[:limit]


# Request tracing - correlation ID support
_REQUEST_ID_HEADER = "X-Request-ID"


def _record_endpoint_metric(endpoint: str, duration_ms: float, status: int, error: bool = False) -> None:
    """Record per-endpoint latency and count metrics."""
    if endpoint not in _endpoint_metrics:
        _endpoint_metrics[endpoint] = {
            "request_count": 0,
            "error_count": 0,
            "total_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "min_latency_ms": float("inf"),
        }
    m = _endpoint_metrics[endpoint]
    m["request_count"] += 1
    m["total_latency_ms"] += duration_ms
    m["max_latency_ms"] = max(m["max_latency_ms"], duration_ms)
    m["min_latency_ms"] = min(m["min_latency_ms"], duration_ms)
    if error:
        m["error_count"] += 1


def _get_endpoint_metrics() -> Dict[str, Any]:
    """Get aggregated endpoint metrics with computed averages."""
    result = {}
    for ep, m in _endpoint_metrics.items():
        count = m["request_count"]
        result[ep] = {
            "request_count": count,
            "error_count": m["error_count"],
            "error_rate": round(m["error_count"] / count, 4) if count > 0 else 0.0,
            "avg_latency_ms": round(m["total_latency_ms"] / count, 2) if count > 0 else 0.0,
            "max_latency_ms": round(m["max_latency_ms"], 2),
            "min_latency_ms": round(m["min_latency_ms"], 2) if m["min_latency_ms"] != float("inf") else 0.0,
        }
    return result


# --------------------------------------------------------------------------- #
# Token usage tracking                                                        #
# --------------------------------------------------------------------------- #

async def _record_token_usage(api_key_hash: Optional[str], prompt_tokens: int,
                              completion_tokens: int) -> None:
    """Record token usage per API key hash."""
    if not api_key_hash:
        return
    async with _token_usage_lock:
        if api_key_hash not in _token_usage:
            _token_usage[api_key_hash] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "request_count": 0,
            }
        usage = _token_usage[api_key_hash]
        usage["prompt_tokens"] += prompt_tokens
        usage["completion_tokens"] += completion_tokens
        usage["total_tokens"] += prompt_tokens + completion_tokens
        usage["request_count"] += 1


async def _check_token_quota(api_key_hash: Optional[str], estimated_tokens: int) -> None:
    """Enforce per-key token quotas. Raises HTTPException if quota would be exceeded."""
    if not api_key_hash:
        return
    max_tokens = _token_quotas.get(api_key_hash, 0)
    if max_tokens <= 0:
        return  # Unlimited
    async with _token_usage_lock:
        usage = _token_usage.get(api_key_hash, {})
        current = usage.get("total_tokens", 0)
        if current + estimated_tokens > max_tokens:
            raise HTTPException(
                status_code=429,
                detail=f"Token quota exceeded: {current}/{max_tokens} tokens used",
                headers={"Retry-After": "3600"},
            )


def get_token_usage() -> Dict[str, Any]:
    """Get a copy of current token usage (hashes truncated for safety)."""
    result = {}
    for kh, usage in _token_usage.items():
        result[kh[:8] + "..."] = dict(usage)
    return result


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = f"{SERVER_NAME}"
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


# --------------------------------------------------------------------------- #
# OpenAPI response models                                                     #
# --------------------------------------------------------------------------- #

class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    uptime_seconds: Optional[float] = None
    active_slots: int = 0
    free_slots: int = 0
    avg_decode_ms: float = 0.0
    port: int
    tls: Dict[str, Any] = {}


class UsageResponse(BaseModel):
    server: str
    provider: str
    total_keys: int
    usage: Dict[str, Any]


# --------------------------------------------------------------------------- #
# FastAPI app with OpenAPI tags                                               #
# --------------------------------------------------------------------------- #

def load_model():
    """Load tokenizer and ONNX model, initialize continuous batcher."""
    global tokenizer, batcher, metrics, CONV_LAYERS, ATTN_LAYERS, HIDDEN, CONV_L_CACHE
    global _request_queue, _shutdown_event, _pending, _active, START_TIME

    _pending = []
    _active = []

    # Record start time on first load
    if START_TIME is None:
        START_TIME = time.time()

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    CONV_LAYERS, ATTN_LAYERS, HIDDEN, CONV_L_CACHE = _load_layer_types(MODEL_PATH)

    batcher = ContinuousBatcher(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        config=config,
        max_batch_size=MAX_BATCH_SIZE,
    )
    batcher.load(provider=PROVIDER)

    # For VitisAI EP, register custom ops (already done in load() if provider is VitisAI)
    if PROVIDER == "VitisAIExecutionProvider" and CUSTOM_OPS_PATH:
        so = batcher._configure_session_options(PROVIDER)
        so.register_custom_ops_library(CUSTOM_OPS_PATH)
        batcher.sess = ort.InferenceSession(
            MODEL_PATH,
            sess_options=so,
            providers=[PROVIDER],
        )
        batcher._output_names = [o.name for o in batcher.sess.get_outputs()]

    input_names = {inp.name for inp in batcher.sess.get_inputs()}
    batcher._needs_past_conv = any("past_conv." in name for name in input_names)
    batcher._needs_past_kv = any("past_key_values" in name for name in input_names)

    # Warmup: run a dummy inference to load model into memory/CPU caches
    if WARMUP_ON_START:
        print(f"{SERVER_NAME}: Warming up model...", flush=True)
        t0 = time.perf_counter()
        batcher.warmup(prompt=WARMUP_PROMPT, max_tokens=WARMUP_TOKENS)
        t1 = time.perf_counter()
        print(f"{SERVER_NAME}: Warmup complete in {(t1-t0)*1000:.0f}ms", flush=True)

    metrics = MetricsCollector(provider=PROVIDER, model=f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}")
    _request_queue = asyncio.Queue()
    _shutdown_event = asyncio.Event()

    # Initialize OpenTelemetry tracer
    otlp_endpoint = os.environ.get("LFM3_OTLP_ENDPOINT")
    tracing_enabled = os.environ.get("LFM3_TRACING_ENABLED", "false").lower() == "true"
    if tracing_enabled and otlp_endpoint:
        init_tracer(service_name="lfm3-cpu", otlp_endpoint=otlp_endpoint, enabled=True)
        print(f"{SERVER_NAME} Server: OpenTelemetry tracing enabled (endpoint: {otlp_endpoint})", flush=True)
    else:
        init_tracer(service_name="lfm3-cpu", enabled=False)

    print(f"{SERVER_NAME} Server: Providers: {batcher.sess.get_providers()}", flush=True)
    print(f"{SERVER_NAME} Server: Model loaded on {PROVIDER}", flush=True)
    print(f"{SERVER_NAME} Server: Conv layers: {CONV_LAYERS}, Attention layers: {ATTN_LAYERS}", flush=True)
    print(f"{SERVER_NAME} Server: max_batch_size={MAX_BATCH_SIZE}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    _shutdown_event_loop = loop

    # Signal handlers for graceful shutdown (SIGTERM / SIGINT)
    def _request_shutdown(sig):
        global _shutting_down
        if not _shutting_down:
            print(f"{SERVER_NAME}: Received signal {sig}, initiating graceful shutdown...", flush=True)
            _shutting_down = True
            # Schedule shutdown on the event loop
            loop.call_soon_threadsafe(lambda: asyncio.create_task(_graceful_shutdown()))

    try:
        loop.add_signal_handler(signal.SIGTERM, _request_shutdown, "SIGTERM")
        loop.add_signal_handler(signal.SIGINT, _request_shutdown, "SIGINT")
    except (NotImplementedError, OSError):
        # Signal handlers not available (e.g., on Windows or non-main thread)
        pass

    load_model()
    task = asyncio.create_task(_batch_loop())
    try:
        yield
    finally:
        await _graceful_shutdown()


# FastAPI app with OpenAPI tags (created after lifespan so it can reference it)
app = FastAPI(
    title=f"LFM3-1.7B Batched Inference Server ({PROVIDER})",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "inference", "description": "Chat completion and batch inference endpoints"},
        {"name": "health", "description": "Health, liveness, and readiness probes"},
        {"name": "observability", "description": "Metrics, SLO, traces, and usage"},
        {"name": "admin", "description": "Administration: shutdown, version, quotas"},
    ],
)


async def _graceful_shutdown():
    """Drain in-flight requests, then stop the batch loop."""
    global _shutting_down
    _shutting_down = True
    print(f"{SERVER_NAME}: Graceful shutdown initiated - stopping batch loop...", flush=True)
    _shutdown_event.set()
    if _batch_task is not None:
        _batch_task.cancel()
        try:
            await _batch_task
        except asyncio.CancelledError:
            pass
    # Wait for active requests to complete (up to 30s)
    deadline = time.monotonic() + 30.0
    while _active and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    print(f"{SERVER_NAME}: Graceful shutdown complete ({len(_active)} active requests drained)", flush=True)


# Enterprise: security headers on every response
app.add_middleware(SecurityHeadersMiddleware)


# Enterprise: ValidationError exception handler -> structured HTTP 400
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content=validation_error_response(exc),
    )


# ====================================================================== #
# Prompt building
# ====================================================================== #
def _build_prompt(messages: List[ChatMessage]) -> str:
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"system: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"user: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"assistant: {msg.content}\n"
    prompt += "assistant: "
    return prompt


# ====================================================================== #
# Background batch loop - event-driven, no sleep(0) hammer
# ====================================================================== #
async def _batch_loop():
    """Background loop: collect requests, batched prefill, batched decode.

    Uses asyncio.Queue with short timeouts for responsive request handling.
    No sleep(0) hammer - yields control only when idle.
    """
    print(f"{SERVER_NAME}: batch loop started", flush=True)

    global _pending, _active

    result_queues: Dict[str, asyncio.Queue] = {}
    req_meta: Dict[str, Dict] = {}

    while not _shutdown_event.is_set():
        try:
            # --- Collect new requests ---
            # Drain all available requests from the queue for batching
            new_items = []
            item = None
            try:
                item = await asyncio.wait_for(_request_queue.get(), timeout=0.01)
                new_items.append(item)
                # Drain any additional requests already in queue
                while True:
                    try:
                        item = _request_queue.get_nowait()
                        new_items.append(item)
                    except asyncio.QueueEmpty:
                        break
            except asyncio.TimeoutError:
                pass

            for item in new_items:
                req_id, prompt, temperature, top_p, max_tokens, result_q = item
                input_ids = tokenizer.encode(prompt).ids
                req = ReqState(req_id, prompt, input_ids, -1)
                req.meta = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                _pending.append(req)
                result_queues[req_id] = result_q
                req_meta[req_id] = req.meta

            # --- Fast path: single pending request, no active requests ---
            # Bypass batch loop overhead for minimum latency on isolated requests
            if len(_pending) == 1 and not _active and batcher._free_slots:
                req = _pending.pop(0)
                slot = batcher._free_slots.pop(0)
                req.slot = slot
                batcher._slots[slot] = req

                result = await batcher.fast_path_generate_single(req)
                req_id, next_token, finished, finish_reason, ttft = result

                _request_metrics[req_id] = {
                    "ttft": ttft,
                    "tokens": 0,
                    "token_times": [],
                    "finished": finished,
                }

                if finished:
                    batcher.finish_request(req, finish_reason or "stop")
                    await result_queues[req_id].put({
                        "token": tokenizer.decode([next_token]),
                        "finished": True,
                        "finish_reason": finish_reason or "stop",
                        "ttft": ttft,
                    })
                else:
                    _active.append(req)
                    token_text = tokenizer.decode([next_token])
                    _request_metrics[req_id]["tokens"] += 1
                    await result_queues[req_id].put({
                        "token": token_text,
                        "finished": False,
                        "finish_reason": None,
                        "ttft": ttft,
                    })

            # --- Prefill new requests (batched) ---
            elif _pending and batcher._free_slots:
                num_slots = len(batcher._free_slots)
                to_prefill = _pending[:num_slots]
                _pending = _pending[num_slots:]

                prefill_results = await batcher.prefill_batch(to_prefill)
                for req, (req_id, logits) in zip(to_prefill, prefill_results):
                    temp = req_meta[req_id]["temperature"]
                    top_p = req_meta[req_id]["top_p"]
                    next_token = _sample_token(logits, temp, top_p)
                    req.next_token = next_token
                    req.generated_ids.append(next_token)
                    req.first_token_time = time.perf_counter()

                    _request_metrics[req_id] = {
                        "ttft": time.perf_counter() - req.prefill_start if req.prefill_start else 0.0,
                        "tokens": 0,
                        "token_times": [],
                        "finished": False,
                    }

                    if next_token == EOS_TOKEN_ID:
                        req.finished = True
                        req.finished_reason = "stop"
                        batcher.finish_request(req, "stop")
                    else:
                        _active.append(req)

                    token_text = tokenizer.decode([next_token])
                    _request_metrics[req_id]["tokens"] += 1
                    await result_queues[req_id].put({
                        "token": token_text,
                        "finished": req.finished,
                        "finish_reason": req.finished_reason,
                        "ttft": time.perf_counter() - req.prefill_start if req.prefill_start else 0.0,
                    })

            # --- Decode active requests (batched with KV cache padding) ---
            if _active:
                still_active: List[ReqState] = []
                for req in _active:
                    req.temperature = req_meta[req.req_id]["temperature"]
                    req.top_p = req_meta[req.req_id]["top_p"]
                    req.max_tokens = req_meta[req.req_id]["max_tokens"]

                # Use batched decode with padding when multiple active requests,
                # single-request fast path decode when only one
                if len(_active) == 1:
                    req = _active[0]
                    result = await batcher.fast_path_decode_single(req)
                    req_id, next_token, finished, finish_reason, token_time = result
                    token_text = tokenizer.decode([next_token])

                    if req_id in _request_metrics:
                        _request_metrics[req_id]["tokens"] += 1
                        _request_metrics[req_id]["token_times"].append(token_time)

                    if finished:
                        batcher.finish_request(req, finish_reason or "stop")
                        _request_metrics[req_id]["finished"] = True
                    else:
                        still_active.append(req)

                    await result_queues[req_id].put({
                        "token": token_text,
                        "finished": finished,
                        "finish_reason": finish_reason,
                    })
                else:
                    decode_results = await batcher.decode_batched(_active)
                    for req, (req_id, next_token, finished, finish_reason, token_time) in zip(_active, decode_results):
                        token_text = tokenizer.decode([next_token])

                        if req_id in _request_metrics:
                            _request_metrics[req_id]["tokens"] += 1
                            _request_metrics[req_id]["token_times"].append(token_time)

                        if finished:
                            batcher.finish_request(req, finish_reason or "stop")
                            _request_metrics[req_id]["finished"] = True
                        else:
                            still_active.append(req)

                        await result_queues[req_id].put({
                            "token": token_text,
                            "finished": finished,
                            "finish_reason": finish_reason,
                        })

                _active = still_active

            # Yield control only when no work was done this iteration
            if item is None and not _pending and not _active:
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"{SERVER_NAME}: batch loop error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            await asyncio.sleep(0.1)


# ====================================================================== #
# API Key authentication (optional - if LFM3_API_KEY is set, all endpoints
# require an X-API-Key header)
# ====================================================================== #
API_KEY = os.environ.get("LFM3_API_KEY", "")


async def verify_api_key(request: Request):
    """Validate API key using RBAC authentication."""
    key_info = _rbac_auth.authenticate(
        request.headers.get("authorization", "").replace("Bearer ", "", 1)
        if request.headers.get("authorization", "").lower().startswith("bearer ")
        else request.headers.get("x-api-key")
    )
    if key_info is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    request.state.api_key_info = key_info
    request.state.api_key_label = key_info.label
    request.state.api_key_hash = key_info.key_hash
    request.state.api_key_role = key_info.role


async def check_rate_limit(request: Request):
    """Token-bucket rate limit dependency.

    Limits per API key when LFM3_API_KEY is configured, otherwise per client IP.
    """
    # label=None => RateLimiter buckets by client IP; label=API_KEY => per key
    retry_after = _rate_limiter.allow(request, API_KEY if API_KEY else None)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded; retry after {retry_after:.1f}s",
            headers={"Retry-After": f"{retry_after:.1f}"},
        )


# ====================================================================== #
# Endpoints
# ====================================================================== #
@app.get("/health")
async def health():
    uptime = None
    if START_TIME is not None:
        uptime = time.time() - START_TIME
    batcher_metrics = batcher.get_metrics() if batcher else {}
    return {
        "status": "healthy",
        "provider": PROVIDER,
        "model": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
        "uptime_seconds": round(uptime, 1) if uptime is not None else None,
        "active_slots": batcher_metrics.get("active_slots", 0),
        "free_slots": batcher_metrics.get("free_slots", 0),
        "avg_decode_ms": batcher_metrics.get("avg_decode_ms", 0),
        "port": SERVER_PORT,
        "tls": tls_status(),
    }


@app.get("/live")
async def liveness():
    """Kubernetes-style liveness probe: the process is alive."""
    return {"status": "alive", "server": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}", "port": SERVER_PORT}


@app.get("/ready")
async def readiness():
    """Kubernetes-style readiness probe: the model/batcher is loaded and serving."""
    if batcher is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _shutting_down:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    batcher_metrics = batcher.get_metrics() if batcher else {}
    return {
        "status": "ready",
        "provider": PROVIDER,
        "model": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
        "port": SERVER_PORT,
        "max_batch_size": batcher.max_batch_size,
        "active_slots": batcher_metrics.get("active_slots", 0),
        "free_slots": batcher_metrics.get("free_slots", 0),
    }


@app.get("/slo")
async def slo_report(request: Request, _=Depends(verify_api_key)):
    """Service-Level Objective report: availability, latency p95, error budget."""
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "read", "/slo")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return _slo_tracker.snapshot()


@app.get("/version")
async def version(request: Request, _=Depends(verify_api_key)):
    """Version info for the server."""
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "read", "/version")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return {
        "server": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
        "provider": PROVIDER,
        "model": "LFM3-1.7B",
        "version": "1.0.0",
        "build": "enterprise",
    }


@app.post("/shutdown")
async def shutdown(request: Request, _=Depends(verify_api_key)):
    """Initiate graceful shutdown (drain in-flight requests, then stop)."""
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "admin", "/shutdown")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    global _shutting_down
    _shutting_down = True
    return {"status": "shutting_down", "server": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [ModelInfo(id=f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}", created=int(time.time())).model_dump()]
    }


@app.get("/v1/model/info")
async def model_info():
    return {
        "id": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
        "provider": PROVIDER,
        "device": "AMD XDNA2 NPU" if PROVIDER == "VitisAIExecutionProvider" else "CPU",
        "model_type": "LFM3-1.7B",
        "conv_layers": "full (original)" if PROVIDER == "CPUExecutionProvider" else "center-tap (Mul approximation)",
        "max_context_length": 262144,
        "vocab_size": 128000,
    }


@app.get("/metrics")
async def metrics_endpoint(request: Request, _=Depends(verify_api_key)):
    if metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not initialized")
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "read", "/metrics")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    batcher_metrics = batcher.get_metrics() if batcher else {}
    prometheus_text = metrics.get_metrics()
    if isinstance(prometheus_text, bytes):
        prometheus_text = prometheus_text.decode("utf-8")
    text = prometheus_text
    text += "\n# Batcher metrics\n"
    for k, v in batcher_metrics.items():
        text += f"{k} {v}\n"
    text += "\n# SLO metrics\n"
    text += prometheus_metrics_all()
    text += "\n# Circuit breaker metrics\n"
    text += circuit_breaker_prometheus_metrics()
    return PlainTextResponse(content=text, media_type=metrics.get_content_type())


@app.get("/v1/batch/status")
async def batch_status(request: Request, _=Depends(verify_api_key)):
    """Return continuous batcher status."""
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "read", "/v1/batch/status")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if batcher is None:
        raise HTTPException(status_code=503, detail="Batcher not initialized")
    return {
        "max_batch_size": batcher.max_batch_size,
        "active_slots": len(batcher.get_active_requests()),
        "free_slots": len(batcher._free_slots),
        "pending": len(_pending),
        "active": len(_active),
        "metrics": batcher.get_metrics(),
    }


@app.get("/v1/metrics/detailed")
async def detailed_metrics(request: Request, _=Depends(verify_api_key)):
    """Return detailed per-request metrics."""
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "read", "/v1/metrics/detailed")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if batcher is None:
        raise HTTPException(status_code=503, detail="Batcher not initialized")

    # Compute summary stats from completed requests
    completed = [m for m in _request_metrics.values() if m["finished"]]
    active_reqs = [m for m in _request_metrics.values() if not m["finished"]]

    def avg_token_time(tokens_list):
        if not tokens_list:
            return 0.0
        return sum(tokens_list) / len(tokens_list)

    summary = {
        "total_requests": len(_request_metrics),
        "completed_requests": len(completed),
        "active_requests": len(active_reqs),
        "pending": len(_pending),
        "batcher_metrics": batcher.get_metrics(),
    }

    if completed:
        ttfts = [m["ttft"] for m in completed if m["ttft"] > 0]
        token_counts = [m["tokens"] for m in completed]
        all_token_times = []
        for m in completed:
            all_token_times.extend(m["token_times"])

        summary["completed"] = {
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts) * 1000, 2) if ttfts else 0,
            "p95_ttft_ms": round(sorted(ttfts)[int(len(ttfts) * 0.95)] * 1000, 2) if ttfts else 0,
            "avg_tokens": round(sum(token_counts) / len(token_counts), 2),
            "avg_token_latency_ms": round(avg_token_time(all_token_times) * 1000, 2) if all_token_times else 0,
        }

    # Recent active request details
    summary["recent_active"] = [
        {
            "req_id": rid,
            "tokens": m["tokens"],
            "ttft_ms": round(m["ttft"] * 1000, 2) if m["ttft"] > 0 else None,
        }
        for rid, m in list(_request_metrics.items())[-10:]
        if not m["finished"]
    ]

    return summary


@app.get("/v1/traces")
async def trace_history(
    request: Request, _=Depends(verify_api_key),
    limit: int = 50,
    provider: Optional[str] = None,
    error_only: bool = False,
):
    """Return recent trace history for this server.

    Query parameters:
    - limit: Max number of traces to return (default 50, max 500)
    - provider: Filter by provider (CPUExecutionProvider or VitisAIExecutionProvider)
    - error_only: Only return requests with errors
    """
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "read", "/v1/traces")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    limit = min(max(limit, 1), 500)
    traces = await _get_trace_history(limit=limit, provider=provider, error_only=error_only)
    return {
        "provider": PROVIDER,
        "port": SERVER_PORT,
        "total_traces": len(_trace_history),
        "returned": len(traces),
        "traces": traces,
    }


@app.get("/v1/metrics/endpoints")
async def endpoint_metrics(request: Request, _=Depends(verify_api_key)):
    """Return per-endpoint request metrics (count, latency, error rate)."""
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "read", "/v1/metrics/endpoints")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return _get_endpoint_metrics()


@app.get("/v1/fleet/status")
async def fleet_status(request: Request, _=Depends(verify_api_key)):
    """Return fleet-compatible status for monitoring aggregation.

    Returns server health, performance metrics, and capacity info
    suitable for consumption by fleet monitoring systems.
    """
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "read", "/v1/fleet/status")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if batcher is None:
        raise HTTPException(status_code=503, detail="Batcher not initialized")

    batcher_metrics = batcher.get_metrics()
    uptime = None
    if START_TIME is not None:
        uptime = time.time() - START_TIME

    # Compute error rate from request metrics
    total_reqs = len(_request_metrics)
    if total_reqs > 0:
        error_count = sum(
            1 for m in _request_metrics.values()
            if m["finished"] and m.get("error")
        )
        error_rate = error_count / total_reqs
    else:
        error_rate = 0.0

    return {
        "server": {
            "name": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
            "provider": PROVIDER,
            "model": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
            "port": SERVER_PORT,
            "uptime_seconds": round(uptime, 1) if uptime is not None else None,
            "status": "healthy",
        },
        "capacity": {
            "max_batch_size": batcher.max_batch_size,
            "active_slots": batcher_metrics.get("active_slots", 0),
            "free_slots": batcher_metrics.get("free_slots", 0),
            "pending_requests": len(_pending),
            "active_requests": len(_active),
        },
        "performance": {
            "avg_prefill_ms": batcher_metrics.get("avg_prefill_ms", 0),
            "avg_decode_ms": batcher_metrics.get("avg_decode_ms", 0),
            "avg_token_latency_ms": batcher_metrics.get("avg_decode_ms", 0),
            "total_tokens_generated": batcher_metrics.get("total_tokens_generated", 0),
            "fast_path_count": batcher_metrics.get("fast_path_count", 0),
            "batched_decode_count": batcher_metrics.get("batched_decode_count", 0),
        },
        "reliability": {
            "total_requests": total_reqs,
            "error_rate": round(error_rate, 4),
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, req: ChatCompletionRequest,
    _=Depends(verify_api_key), __=Depends(check_rate_limit)
):
    try:
        return await _chat_completions_impl(request, req)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"CHAT_COMPLETIONS ERROR: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


async def _chat_completions_impl(
    request: Request, req: ChatCompletionRequest,
    _=Depends(verify_api_key), __=Depends(check_rate_limit)
):

    # Enterprise: graceful shutdown - reject new requests while draining
    if _shutting_down:
        raise HTTPException(status_code=503, detail="Server is shutting down, try again later")

    # Enterprise: load shedding - reject when overloaded
    active_count = len(_active) + len(_pending)
    if active_count >= _load_shed_max_active:
        _audit.rate_limit_exceeded(
            actor=getattr(request.state, "api_key_hash", None) or (request.client.host if request.client else "unknown"),
            actor_type="api_key",
            request_id=getattr(request.state, "request_id", None),
        )
        raise HTTPException(
            status_code=503,
            detail="Server overloaded; try again later",
            headers={"Retry-After": "5"},
        )

    tracer = get_tracer()

    client_ip = request.client.host if request.client else "unknown"
    request_id = request.headers.get("X-Request-ID") or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    start_time = time.perf_counter()

    # Enterprise: W3C trace context propagation - extract incoming traceparent
    incoming_traceparent = request.headers.get("traceparent")

    # Enterprise: input validation
    api_key_label = getattr(request.state, "api_key_label", None)
    api_key_hash = getattr(request.state, "api_key_hash", None)
    api_key_role = getattr(request.state, "api_key_role", None)
    try:
        validate_client(api_key_label or client_ip)
        validate_messages(req.messages)
        validate_max_tokens(req.max_tokens)
    except ValidationError as e:
        log_event(
            "validation.rejected",
            request_id=request_id,
            client_ip=client_ip,
            endpoint="/v1/chat/completions",
            method="POST",
            status_code=400,
            extra={"code": e.code},
        )
        raise HTTPException(status_code=400, detail=f"{e.code}: {e.message}")

    # Enterprise: audit - authentication success
    if api_key_label:
        _audit.auth_success(actor=api_key_hash or client_ip, actor_type="api_key", request_id=request_id)

    # Enterprise: RBAC permission check - write access required for inference
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "write", "/v1/chat/completions")
    except PermissionError as e:
        _audit.authz_denied(
            actor=api_key_hash or client_ip,
            resource="/v1/chat/completions",
            request_id=request_id,
        )
        log_event(
            "authz.denied",
            request_id=request_id,
            client_ip=client_ip,
            endpoint="/v1/chat/completions",
            method="POST",
            status_code=403,
            extra={"required": "write", "role": api_key_role},
        )
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"RBAC ERROR: {e}\n{tb}", flush=True)
        _audit.authz_denied(
            actor=api_key_hash or client_ip,
            resource="/v1/chat/completions",
            request_id=request_id,
        )
        log_event(
            "authz.error",
            request_id=request_id,
            client_ip=client_ip,
            endpoint="/v1/chat/completions",
            method="POST",
            status_code=500,
            extra={"error": str(e), "role": api_key_role},
        )
        raise HTTPException(status_code=500, detail=f"RBAC error: {e}")

    # Enterprise: token quota check - reject if per-key quota would be exceeded
    estimated_prompt_tokens = len(tokenizer.encode(prompt).ids) if tokenizer else 0
    await _check_token_quota(api_key_hash, estimated_prompt_tokens + (req.max_tokens or 50))

    with trace_request(
        tracer, PROVIDER, f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
        "chat/completions", 0, req.max_tokens or 50
    ) as request_span:
        if incoming_traceparent:
            request_span.set_attribute("w3c.traceparent", incoming_traceparent)
        endpoint = "chat/completions"
        start_time = time.perf_counter()

        # Request ID tracing - use client-provided X-Request-ID or generate one
        client_request_id = request.headers.get(_REQUEST_ID_HEADER)
        req_id = client_request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"

        prompt = _build_prompt(req.messages)
        max_tokens = req.max_tokens or 50
        temperature = req.temperature or 0.7
        top_p = req.top_p or 1.0

        request_span.set_attribute("llm.prompt_tokens", len(tokenizer.encode(prompt).ids))

        result_queue = asyncio.Queue()

        await _request_queue.put((req_id, prompt, temperature, top_p, max_tokens, result_queue))

        if req.stream:
            return StreamingResponse(
                _stream_results(req_id, result_queue, max_tokens),
                media_type="text/event-stream",
            )

        generated_text = ""
        finish_reason = "stop"
        http_error = None
        # Enterprise: audit - inference started
        _audit.inference_started(
            actor=api_key_hash or client_ip,
            resource="chat/completions",
            request_id=request_id,
        )
        try:
            with _circuit_breaker.call("cpu_inference"):
                while True:
                    try:
                        result = await asyncio.wait_for(result_queue.get(), timeout=120.0)
                    except asyncio.TimeoutError:
                        http_error = HTTPException(status_code=504, detail="Generation timeout")
                        break

                    generated_text += result.get("token", "")
                    if result.get("finished"):
                        finish_reason = result.get("finish_reason", "stop")
                        break
        except CircuitOpenError as e:
            record_request(PROVIDER, f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
                           False, (time.perf_counter() - start_time) * 1000, "chat/completions")
            _audit.circuit_breaker_open(
                resource="cpu_inference",
                retry_after=e.retry_after,
                request_id=request_id,
            )
            _audit.inference_failed(
                actor=api_key_hash or client_ip,
                resource="cpu_inference",
                error="circuit_open",
                request_id=request_id,
            )
            log_event(
                "inference.circuit_open",
                request_id=request_id,
                client_ip=client_ip,
                endpoint="/v1/chat/completions",
                method="POST",
                status_code=503,
                provider=PROVIDER,
                model=f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
            )
            metrics.record_error("chat/completions")
            raise HTTPException(
                status_code=503,
                detail=f"Inference circuit open; retry after {e.retry_after:.1f}s",
            )
        if http_error is not None:
            raise http_error

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Record per-endpoint metrics
        _record_endpoint_metric(endpoint, duration_ms, 200, error=False)

        # Record in metrics collector
        if metrics:
            completion_tokens = len(tokenizer.encode(generated_text).ids)
            metrics.record_decode(duration_ms / 1000.0, completion_tokens)

        # Record trace for fleet querying
        ttft_ms = _request_metrics.get(req_id, {}).get("ttft", 0) * 1000
        await _record_trace(
            req_id=req_id,
            prompt=prompt[:200],
            prompt_tokens=len(tokenizer.encode(prompt).ids),
            completion_tokens=len(tokenizer.encode(generated_text).ids),
            total_tokens=len(tokenizer.encode(prompt).ids) + len(tokenizer.encode(generated_text).ids),
            ttft_ms=ttft_ms,
            total_latency_ms=duration_ms,
            finish_reason=finish_reason,
            provider=PROVIDER,
            temperature=temperature,
            top_p=top_p,
            error=False,
        )

        request_span.set_attribute("llm.completion_tokens", len(tokenizer.encode(generated_text).ids))
        request_span.set_attribute("llm.duration_seconds", duration_ms / 1000.0)

        # Enterprise: SLO tracking - record successful request
        record_request(PROVIDER, f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
                       True, duration_ms, "chat/completions")

        # Enterprise: token usage tracking per API key
        completion_toks = len(tokenizer.encode(generated_text).ids)
        prompt_toks = len(tokenizer.encode(prompt).ids)
        await _record_token_usage(api_key_hash, prompt_toks, completion_toks)

        # Enterprise: audit - inference completed
        _audit.inference_completed(
            actor=api_key_hash or client_ip,
            resource="chat/completions",
            request_id=request_id,
            latency_ms=duration_ms,
        )

        # Log request
        _log("INFO", "chat_completions",
             req_id=req_id,
             duration_ms=round(duration_ms, 2),
             tokens=len(tokenizer.encode(generated_text).ids),
             finish_reason=finish_reason,
             provider=PROVIDER)

        return {
            "id": req_id,
            "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated_text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt).ids),
            "completion_tokens": len(tokenizer.encode(generated_text).ids),
            "total_tokens": len(tokenizer.encode(prompt).ids) + len(tokenizer.encode(generated_text).ids),
        },
        "request_id": req_id,
    }


@app.post("/v1/batch/chat/completions")
async def batch_chat_completions(
    reqs: List[ChatCompletionRequest],
    _=Depends(verify_api_key), __=Depends(check_rate_limit)
):
    """Batch multiple chat completion requests in a single HTTP call.

    All prompts are prefilled together and decoded together, maximizing
    throughput for concurrent workloads.
    """
    if batcher is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result_queues: Dict[str, asyncio.Queue] = {}
    req_ids: List[str] = []
    prompts: List[str] = []
    temperatures: List[float] = []
    top_ps: List[float] = []
    max_tokens_list: List[int] = []

    for req in reqs:
        prompt = _build_prompt(req.messages)
        max_tokens = req.max_tokens or 50
        temperature = req.temperature or 0.7
        top_p = req.top_p or 1.0

        rq = asyncio.Queue()
        rid = f"batchcmpl-{uuid.uuid4().hex[:12]}"

        result_queues[rid] = rq
        req_ids.append(rid)
        prompts.append(prompt)
        temperatures.append(temperature)
        top_ps.append(top_p)
        max_tokens_list.append(max_tokens)

        await _request_queue.put((rid, prompt, temperature, top_p, max_tokens, rq))

    # Collect results for all requests
    results: Dict[str, str] = {}
    finished: Dict[str, bool] = {rid: False for rid in req_ids}
    finish_reasons: Dict[str, str] = {rid: "stop" for rid in req_ids}

    remaining = set(req_ids)
    while remaining:
        done_this_round = []
        for rid in list(remaining):
            try:
                result = await asyncio.wait_for(result_queues[rid].get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            results[rid] = results.get(rid, "") + result.get("token", "")
            if result.get("finished"):
                finished[rid] = True
                finish_reasons[rid] = result.get("finish_reason", "stop")
                done_this_round.append(rid)

        for rid in done_this_round:
            remaining.discard(rid)

    return {
        "results": [
            {
                "id": rid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": results.get(rid, "")},
                    "finish_reason": finish_reasons.get(rid, "stop"),
                }],
                "usage": {
                    "prompt_tokens": len(tokenizer.encode(prompts[req_ids.index(rid)]).ids),
                    "completion_tokens": len(tokenizer.encode(results.get(rid, "")).ids),
                    "total_tokens": len(tokenizer.encode(prompts[req_ids.index(rid)]).ids)
                    + len(tokenizer.encode(results.get(rid, "")).ids),
                },
            }
            for rid in req_ids
        ]
    }


async def _stream_results(req_id: str, result_queue: asyncio.Queue, max_tokens: int):
    """Stream SSE results for a request."""
    chunk_id = req_id
    created = int(time.time())
    tokens_received = 0

    try:
        while tokens_received < max_tokens:
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                break

            token = result.get("token", "")
            finished = result.get("finished", False)
            finish_reason = result.get("finish_reason", "stop")

            if token:
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                tokens_received += 1

            if finished:
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield f"data: [DONE]\n\n"
                break
    except Exception as e:
        error_chunk = {"error": {"message": str(e), "type": "runtime_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


# ====================================================================== #
# WebSocket streaming endpoint                                           #
# ====================================================================== #

@app.websocket("/v1/stream/chat/completions")
async def websocket_chat_completions(websocket: WebSocket):
    """WebSocket endpoint for real-time token streaming.

    Client sends a JSON chat completion request, receives tokens one at a
    time as JSON messages, then a final completion message.
    """
    await websocket.accept()
    key_info = getattr(websocket.scope.get("state", {}), "api_key_info", None) if hasattr(websocket.scope.get("state", {}), "api_key_info") else None
    api_key_hash = None
    api_key_label = None
    try:
        # First message must be the request
        raw = await websocket.receive_text()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_json({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}})
            await websocket.close(code=4000)
            return

        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 50)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 1.0)
        model = data.get("model")

        # Basic validation
        if not messages:
            await websocket.send_json({"error": {"message": "messages is required", "type": "invalid_request_error"}})
            await websocket.close(code=4000)
            return

        # Build prompt
        prompt = _build_prompt([ChatMessage(role=m.get("role", "user"), content=m.get("content", "")) for m in messages])

        # Token quota check
        if tokenizer:
            estimated = len(tokenizer.encode(prompt).ids) + max_tokens
            await _check_token_quota(api_key_hash, estimated)

        result_queue = asyncio.Queue()
        req_id = f"ws-{uuid.uuid4().hex[:12]}"
        await _request_queue.put((req_id, prompt, temperature, top_p, max_tokens, result_queue))

        tokens_received = 0
        try:
            while tokens_received < max_tokens:
                try:
                    result = await asyncio.wait_for(result_queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    await websocket.send_json({"error": {"message": "Generation timeout", "type": "runtime_error"}})
                    break

                token = result.get("token", "")
                finished = result.get("finished", False)
                finish_reason = result.get("finish_reason", "stop")

                if token:
                    await websocket.send_json({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                    })
                    tokens_received += 1

                if finished:
                    await websocket.send_json({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    })
                    # Record token usage
                    if tokenizer:
                        completion_toks = len(tokenizer.encode(token).ids) if token else 0
                        prompt_toks = len(tokenizer.encode(prompt).ids)
                        await _record_token_usage(api_key_hash, prompt_toks, completion_toks)
                    break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"error": {"message": str(e), "type": "runtime_error"}})
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ====================================================================== #
# Token usage endpoint                                                   #
# ====================================================================== #

@app.get("/v1/usage")
async def usage_report(request: Request, _=Depends(verify_api_key)):
    """Return token usage statistics (all keys or current key if not admin)."""
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "read", "/v1/usage")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    usage = get_token_usage()

    # If not admin, only return the caller's own usage
    if key_info and key_info.role != "admin":
        caller_hash = getattr(request.state, "api_key_hash", None)
        if caller_hash:
            usage = {k: v for k, v in usage.items() if k.startswith(caller_hash[:8])}

    return {
        "server": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
        "provider": PROVIDER,
        "total_keys": len(_token_usage),
        "usage": usage,
    }


@app.post("/v1/usage/quota")
async def set_token_quota(request: Request, quota: Dict[str, Any]):
    """Set a token quota for an API key hash.

    Body: {"key_hash": "abc123...", "max_tokens": 1000000}
    Set max_tokens to 0 for unlimited.
    Requires admin role.
    """
    key_info = getattr(request.state, "api_key_info", None)
    try:
        _rbac_auth.require_permission(key_info, "admin", "/v1/usage/quota")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    key_hash = quota.get("key_hash")
    max_tokens = int(quota.get("max_tokens", 0))
    if not key_hash:
        raise HTTPException(status_code=400, detail="key_hash is required")
    _token_quotas[key_hash] = max_tokens
    _log("INFO", "token_quota_set", key_hash=key_hash[:8] + "...", max_tokens=max_tokens)
    return {"status": "ok", "key_hash": key_hash[:8] + "...", "max_tokens": max_tokens}


# ====================================================================== #
# Detailed health endpoint                                               #
# ====================================================================== #

@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check with dependency status (NPU, disk, memory, model)."""
    checks = {}
    overall = "healthy"

    # Model / batcher status
    if batcher is None or tokenizer is None:
        checks["model"] = {"status": "critical", "message": "Model not loaded"}
        overall = "critical"
    else:
        bm = batcher.get_metrics() if batcher else {}
        checks["model"] = {
            "status": "ok",
            "provider": PROVIDER,
            "active_slots": bm.get("active_slots", 0),
            "free_slots": bm.get("free_slots", 0),
        }

    # Uptime
    uptime = None
    if START_TIME is not None:
        uptime = round(time.time() - START_TIME, 1)
    checks["uptime"] = {"status": "ok", "uptime_seconds": uptime}

    # Disk usage
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        disk_pct = round(used / total * 100, 1)
        disk_status = "critical" if disk_pct > 90 else ("warning" if disk_pct > 80 else "ok")
        if disk_status != "ok":
            overall = disk_status
        checks["disk"] = {
            "status": disk_status,
            "total_gb": round(total / (1024 ** 3), 2),
            "used_gb": round(used / (1024 ** 3), 2),
            "free_gb": round(free / (1024 ** 3), 2),
            "usage_percent": disk_pct,
        }
    except Exception as e:
        checks["disk"] = {"status": "unknown", "error": str(e)}

    # Memory usage
    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_status = "critical" if mem.percent > 90 else ("warning" if mem.percent > 80 else "ok")
        if mem_status != "ok":
            overall = mem_status
        checks["memory"] = {
            "status": mem_status,
            "usage_percent": mem.percent,
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "total_gb": round(mem.total / (1024 ** 3), 2),
        }
    except Exception as e:
        checks["memory"] = {"status": "unknown", "error": str(e)}

    # NPU device check (if NPU provider)
    if PROVIDER == "VitisAIExecutionProvider":
        try:
            import os
            xrt_ok = os.path.exists("/opt/xilinx/xrt/lib/libxrt_core.so")
            checks["npu"] = {
                "status": "ok" if xrt_ok else "warning",
                "provider": PROVIDER,
                "xrt_library": xrt_ok,
            }
        except Exception as e:
            checks["npu"] = {"status": "unknown", "error": str(e)}
    else:
        checks["npu"] = {"status": "not_applicable", "provider": PROVIDER}

    # TLS status
    checks["tls"] = tls_status()

    return {
        "status": overall,
        "server": f"{SERVER_NAME}-{PROVIDER_LABEL.lower()}",
        "provider": PROVIDER,
        "port": SERVER_PORT,
        "uptime_seconds": uptime,
        "checks": checks,
    }


if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1238)
    parser.add_argument("--tls", action="store_true", default=None,
                        help="Enable TLS/HTTPS (default: LFM3_TLS_ENABLED env)")
    parser.add_argument("--tls-dir", default=None,
                        help="TLS certificate directory (default: LFM3_TLS_DIR env)")
    parser.add_argument("--tls-gen-cert", action="store_true",
                        help="Generate self-signed dev certificates and exit")
    args = parser.parse_args()

    # Handle certificate generation
    if args.tls_gen_cert:
        ok, msg = generate_dev_certificates(force=True)
        print(msg)
        raise SystemExit(0 if ok else 1)

    # Resolve TLS configuration from flags / env
    tls_enabled = args.tls if args.tls is not None else tls_ready()
    if args.tls_dir:
        import os as _os
        _os.environ["LFM3_TLS_DIR"] = args.tls_dir
        # Re-point module-level paths
        from lfm3_npu import tls as _tls_mod
        _tls_mod.TLS_DIR = Path(args.tls_dir)
        _tls_mod.CA_CERT = _tls_mod.TLS_DIR / "ca.crt"
        _tls_mod.CA_KEY = _tls_mod.TLS_DIR / "ca.key"
        _tls_mod.SERVER_CERT = _tls_mod.TLS_DIR / "server.crt"
        _tls_mod.SERVER_KEY = _tls_mod.TLS_DIR / "server.key"

    ssl_ctx = ssl_context() if tls_enabled else None
    scheme = "https" if ssl_ctx else "http"
    print(f"{SERVER_NAME}: Starting {scheme} server on {args.host}:{args.port}", flush=True)
    if tls_enabled and not ssl_ctx:
        print(f"{SERVER_NAME}: WARNING --tls requested but no cert/key found. "
              f"Run with --tls-gen-cert to generate dev certs.", flush=True)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        server_header=False,
        ssl_certfile=TLS_CERT_FILE if ssl_ctx else None,
        ssl_keyfile=TLS_KEY_FILE if ssl_ctx else None,
        ssl_ca_certs=TLS_CA_FILE if (ssl_ctx and TLS_VERIFY_CLIENT) else None,
        ssl_cert_reqs=ssl.CERT_REQUIRED if (ssl_ctx and TLS_VERIFY_CLIENT) else ssl.CERT_NONE,
    )
