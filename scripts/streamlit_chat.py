"""
LFM3-1.7B NPU Inference - Streamlit Chat UI

A simple chat interface for interacting with the LFM3-1.7B inference servers.
Connects to either the CPU (port 1238) or NPU (port 1239) server via HTTP.

Usage:
    streamlit run scripts/streamlit_chat.py --server.port 8501
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict

# Page config
st.set_page_config(
    page_title="LFM3-1.7B NPU Chat",
    page_icon="🧠",
    layout="wide",
)

# Default server URLs
DEFAULT_CPU_URL = "http://localhost:1238"
DEFAULT_NPU_URL = "http://localhost:1239"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "server_url" not in st.session_state:
    st.session_state.server_url = DEFAULT_NPU_URL
if "server_type" not in st.session_state:
    st.session_state.server_type = "NPU"
if "generation_stats" not in st.session_state:
    st.session_state.generation_stats = {}


def check_server_health(url: str) -> tuple[bool, str]:
    """Check if a server is reachable and healthy."""
    try:
        resp = requests.get(f"{url}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return True, data.get("status", "unknown")
        return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)


def check_models(url: str) -> list:
    """Check available models on a server."""
    try:
        resp = requests.get(f"{url}/v1/models", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return [m.get("id", "unknown") for m in data.get("data", [])]
    except Exception:
        pass
    return []


def generate_response(messages: List[Dict], max_tokens: int, temperature: float,
                      top_p: float, stream: bool = True) -> dict:
    """Send a generation request to the server."""
    url = st.session_state.server_url
    model = "lfm3:1.7b" if st.session_state.server_type == "NPU" else "lfm3-1.7b-cpu"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }

    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        elapsed = time.perf_counter() - start

        if resp.status_code != 200:
            return {
                "error": True,
                "message": f"Server error: HTTP {resp.status_code}",
                "elapsed": elapsed,
            }

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage = data.get("usage", {})

        return {
            "error": False,
            "content": msg.get("content", ""),
            "reasoning": msg.get("reasoning_content", ""),
            "elapsed": elapsed,
            "usage": usage,
            "finish_reason": choice.get("finish_reason", ""),
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "error": True,
            "message": str(e),
            "elapsed": elapsed,
        }


def stream_generate(messages: List[Dict], max_tokens: int, temperature: float,
                    top_p: float):
    """Stream generation response from the server."""
    url = st.session_state.server_url
    model = "lfm3:1.7b" if st.session_state.server_type == "NPU" else "lfm3-1.7b-cpu"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }

    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120,
        )

        if resp.status_code != 200:
            yield {
                "error": True,
                "message": f"Server error: HTTP {resp.status_code}",
                "elapsed": time.perf_counter() - start,
            }
            return

        full_content = ""
        full_reasoning = ""
        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                                yield {
                                    "error": False,
                                    "content": content,
                                    "full_content": full_content,
                                    "reasoning": full_reasoning,
                                }
                    except json.JSONDecodeError:
                        pass

        elapsed = time.perf_counter() - start
        yield {
            "error": False,
            "content": "",
            "full_content": full_content,
            "reasoning": full_reasoning,
            "elapsed": elapsed,
            "done": True,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        yield {
            "error": True,
            "message": str(e),
            "elapsed": elapsed,
        }


# Sidebar configuration
st.sidebar.title("🧠 LFM3-1.7B Settings")

server_type = st.sidebar.selectbox(
    "Inference Server",
    ["NPU", "CPU"],
    index=0 if st.session_state.server_type == "NPU" else 1,
    help="NPU = FastFlowLM on AMD XDNA2 (port 1239), CPU = ONNX Runtime (port 1238)",
)
st.session_state.server_type = server_type

if server_type == "NPU":
    st.session_state.server_url = DEFAULT_NPU_URL
else:
    st.session_state.server_url = DEFAULT_CPU_URL

max_tokens = st.sidebar.slider("Max Tokens", 10, 200, 50, 10)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
top_p = st.sidebar.slider("Top P", 0.1, 1.0, 0.9, 0.1)
stream = st.sidebar.checkbox("Stream Response", value=True)

# Health check
st.sidebar.divider()
st.sidebar.subheader("Server Health")

if st.sidebar.button("Check Health"):
    is_healthy, detail = check_server_health(st.session_state.server_url)
    if is_healthy:
        st.sidebar.success(f"✅ {server_type} server healthy ({detail})")
        models = check_models(st.session_state.server_url)
        if models:
            st.sidebar.info(f"Models: {', '.join(models)}")
    else:
        st.sidebar.error(f"❌ {server_type} server unreachable: {detail}")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.generation_stats = {}
    st.rerun()

# Main chat interface
st.title("🧠 LFM3-1.7B NPU Inference")
st.markdown(f"""
Connected to **{server_type}** server at `{st.session_state.server_url}`.

NPU server runs on AMD XDNA2 Strix Halo via FastFlowLM v1.0.12 — full-quality
inference with reasoning traces (Qwen XML format).
""")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("reasoning"):
            with st.expander("🔍 Reasoning", expanded=False):
                st.markdown(message["reasoning"])
        if message.get("stats"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("TTFT", f"{message['stats'].get('ttft', 0):.2f}s")
            with col2:
                st.metric("Decode", f"{message['stats'].get('decode_tps', 0):.1f} tok/s")
            with col3:
                st.metric("Total", f"{message['stats'].get('elapsed', 0):.2f}s")

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        reasoning_placeholder = st.empty()
        stats_placeholder = st.empty()

        full_content = ""
        full_reasoning = ""
        start_time = time.perf_counter()

        if stream:
            for chunk in stream_generate(
                st.session_state.messages,
                max_tokens,
                temperature,
                top_p,
            ):
                if chunk.get("error"):
                    st.error(f"Generation failed: {chunk['message']}")
                    break
                if chunk.get("done"):
                    elapsed = chunk.get("elapsed", 0)
                    usage = chunk.get("usage", {})
                    stats = {
                        "elapsed": elapsed,
                        "ttft": usage.get("prefill_duration_ttft", 0),
                        "decode_tps": usage.get("decoding_speed_tps", 0),
                    }
                    st.session_state.generation_stats[len(st.session_state.messages)] = stats
                    with stats_placeholder.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("TTFT", f"{stats['ttft']:.2f}s")
                        with col2:
                            st.metric("Decode", f"{stats['decode_tps']:.1f} tok/s")
                        with col3:
                            st.metric("Total", f"{stats['elapsed']:.2f}s")
                    break
                content = chunk.get("content", "")
                full_content += content
                reasoning = chunk.get("reasoning", "")
                if reasoning:
                    full_reasoning = reasoning
                message_placeholder.markdown(full_content + "▌")
                if reasoning:
                    reasoning_placeholder.markdown(f"*{reasoning}*")
        else:
            result = generate_response(
                st.session_state.messages,
                max_tokens,
                temperature,
                top_p,
            )
            if result.get("error"):
                st.error(f"Generation failed: {result['message']}")
            else:
                full_content = result.get("content", "")
                full_reasoning = result.get("reasoning", "")
                elapsed = result.get("elapsed", 0)
                usage = result.get("usage", {})
                stats = {
                    "elapsed": elapsed,
                    "ttft": usage.get("prefill_duration_ttft", 0),
                    "decode_tps": usage.get("decoding_speed_tps", 0),
                }
                st.session_state.generation_stats[len(st.session_state.messages)] = stats
                message_placeholder.markdown(full_content)
                if full_reasoning:
                    reasoning_placeholder.markdown(f"*{full_reasoning}*")
                with stats_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("TTFT", f"{stats['ttft']:.2f}s")
                    with col2:
                        st.metric("Decode", f"{stats['decode_tps']:.1f} tok/s")
                    with col3:
                        st.metric("Total", f"{stats['elapsed']:.2f}s")

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_content,
            "reasoning": full_reasoning,
            "stats": st.session_state.generation_stats.get(len(st.session_state.messages), {}),
        })

# Footer
st.divider()
st.markdown("""
---
**LFM3-1.7B NPU Inference Tooling** — AMD XDNA2 Strix Halo | FastFlowLM v1.0.12

[GitHub](https://github.com/scottjoyner/lfm3-npu-tool) | [Runbook](OPERATIONS_RUNBOOK.md)
""")
