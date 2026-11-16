# Contributing to LFM3-1.7B NPU Inference Tooling

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Getting Started

### Prerequisites
- AMD Strix Halo NPU (aie2p, 6x8) with XRT 2.25.37+
- FastFlowLM v1.0.12+ with `liblfm3_npu.so`
- Python 3.14+ (Ryzen AI SDK requires Python 3.12 in a separate venv)
- No sudo required — all services run as user systemd units

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/scottjoyner/lfm3-npu-tool.git
   cd lfm3-npu-tool
   ```

2. Run the deploy script:
   ```bash
   ./deploy.sh
   ```

3. Verify all services are running:
   ```bash
   lfm3-npu validate
   ```

## Making Changes

### Code Style
- Python: Follow PEP 8, use type hints where practical
- Shell: Use `set -euo pipefail`, quote all variables
- YAML/JSON: Validate before committing

### Testing
Run the test suite before submitting a PR:
```bash
python3 -m pytest tests/test_tooling.py -v
```

Run the CI test suite:
```bash
python3 scripts/ci_test.py
```

### Committing
- Use conventional commit messages (e.g., `feat:`, `fix:`, `docs:`, `refactor:`)
- Keep commits focused on a single change
- Push to a feature branch, then open a pull request

## Reporting Issues

When reporting bugs, please include:
- OS and kernel version
- NPU hardware details (`xrt-smi examine`)
- FastFlowLM version
- Steps to reproduce
- Expected vs. actual behavior
- Relevant logs from `journalctl --user -u lfm3-npu-*`

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
