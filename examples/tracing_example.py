#!/usr/bin/env python3
"""
Example: OpenTelemetry Distributed Tracing
Demonstrates initializing the tracer, creating spans for prefill/decode
phases, and configuring OTLP export to a collector (e.g., Jaeger).
"""
import sys
import os
import time

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))

from lfm3_npu import init_tracer, get_tracer, trace_request, trace_prefill, trace_decode


def main():
    print("=== LFM3-1.7B OpenTelemetry Tracing Example ===\n")

    # Initialize tracer without OTLP endpoint (no-op mode for local testing)
    print("1. Initializing tracer (no OTLP endpoint - no-op mode)...")
    tracer = init_tracer(
        service_name="lfm3-inference-example",
        otlp_endpoint=None,  # Set to "http://localhost:4318" for Jaeger/OTel collector
        enabled=True,
    )
    print(f"   Tracer: {tracer}")

    # Simulate a traced inference request
    print("\n2. Simulating a traced inference request...")
    with trace_request(
        tracer, "cpu", "lfm3-1.7b-cpu",
        "chat/completions", 100, 50
    ) as request_span:
        request_span.set_attribute("user.id", "user-123")
        request_span.set_attribute("request.id", "req-abc-123")

        # Simulate prefill phase
        print("   Running prefill phase...")
        with trace_prefill(
            tracer, "cpu", "lfm3-1.7b-cpu", 100
        ) as prefill_span:
            time.sleep(0.05)  # Simulate work
            prefill_span.set_attribute("tokens.processed", 100)
            prefill_span.set_attribute("kv_cache.size", 2048)

        # Simulate decode phase
        print("   Running decode phase...")
        with trace_decode(
            tracer, "cpu", "lfm3-1.7b-cpu", 1, 50
        ) as decode_span:
            time.sleep(0.1)  # Simulate work
            decode_span.set_attribute("tokens.generated", 50)
            decode_span.set_attribute("tokens.per.second", 500)

    print("   Request trace complete\n")

    # Show how to initialize with OTLP endpoint for production
    print("3. For production, initialize with OTLP endpoint:")
    print('   tracer = init_tracer(')
    print('       service_name="lfm3-inference",')
    print('       otlp_endpoint="http://localhost:4318",')
    print('       enabled=True,')
    print('   )')
    print("   This exports traces to Jaeger/Tempo/OpenTelemetry Collector\n")

    print("✓ OpenTelemetry tracing example complete")


if __name__ == "__main__":
    main()
