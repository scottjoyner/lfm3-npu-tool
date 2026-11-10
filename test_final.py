"""
Final comprehensive test of LFM3-1.7B harness.
"""
import json
import urllib.request
import time

def test_server(port, name):
    print(f"\n=== {name} Server (port {port}) ===")
    
    # Test health
    req = urllib.request.Request(f"http://0.0.0.0:{port}/health")
    d = json.loads(urllib.request.urlopen(req, timeout=5).read())
    print(f"  Status: {d.get('status')}")
    print(f"  Uptime: {d.get('uptime_seconds', 0):.1f}s")
    print(f"  Active slots: {d.get('active_slots', 0)}/{d.get('max_batch_size', '?')}")
    
    # Test generation
    req = urllib.request.Request(
        f"http://0.0.0.0:{port}/v1/chat/completions",
        data=json.dumps({
            "messages": [{"role": "user", "content": "The quick brown fox jumps over the lazy dog"}],
            "max_tokens": 50,
            "stream": False
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    t1 = time.perf_counter()
    
    content = d["choices"][0]["message"]["content"]
    tokens = d["usage"]["completion_tokens"]
    elapsed = t1 - t0
    
    print(f"  Tokens: {tokens}")
    print(f"  Time: {elapsed*1000:.1f}ms")
    print(f"  Tokens/sec: {tokens/elapsed:.1f}")
    print(f"  Content: {repr(content[:80])}")

if __name__ == "__main__":
    test_server(1238, "CPU")
    test_server(1239, "NPU")