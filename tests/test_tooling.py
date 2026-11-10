#!/usr/bin/env python3
"""
Test suite for LFM3-1.7B NPU tooling.
"""

import sys
import os

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))


def test_config_loading():
    """Test that configuration loads correctly."""
    from lfm3_npu import load_config, Config

    config = load_config()
    assert isinstance(config, Config), "Config should be a Config instance"
    assert config.get("model.path") is not None, "model.path should be set"
    assert config.get("npu.provider") == "VitisAIExecutionProvider"
    assert config.get("server.cpu.port") == 1238
    assert config.get("server.npu.port") == 1239
    print("  ✓ Config loading works")


def test_module_imports():
    """Test that all modules can be imported."""
    from lfm3_npu import (
        Config,
        LFM3NPUInference,
        ServerManager,
        bench_model,
        load_config,
        convert_model,
    )
    print("  ✓ All module imports work")


def test_converter():
    """Test that the converter module exists and has expected functions."""
    from lfm3_npu.converter import convert, convert_centertap, convert_bypass

    assert callable(convert)
    assert callable(convert_centertap)
    assert callable(convert_bypass)
    print("  ✓ Converter module works")


def test_benchmark():
    """Test that the benchmark module has expected functions."""
    from lfm3_npu.benchmark import bench_model
    assert callable(bench_model)
    print("  ✓ Benchmark module works")


def test_server_manager():
    """Test that ServerManager can be instantiated."""
    from lfm3_npu import ServerManager, load_config

    config = load_config()
    manager = ServerManager(config)
    assert manager is not None
    print("  ✓ ServerManager instantiation works")


def main():
    """Run all tests."""
    print("Running LFM3-1.7B NPU tooling tests...\n")

    tests = [
        test_config_loading,
        test_module_imports,
        test_converter,
        test_benchmark,
        test_server_manager,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())