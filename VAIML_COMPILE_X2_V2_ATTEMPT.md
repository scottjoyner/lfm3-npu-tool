# vaiml_compile_x2_v2 Attempt — Detailed Findings

## Summary

`libvaiml2.so` exports `vaiml_compile_x2_v2` (a C function, not mangled). Unlike `libvaiml.so` which crashes with SIGSEGV on dlopen, `libvaiml2.so` loads successfully and `vaiml_compile_x2_v2` actually executes (doesn't segfault). However, it consistently fails with:

```
Failed to create directory: ""
Reason: Invalid argument
ERROR: [aiecompiler 77-401] Could not create working dir .
Compilation Failed
```

## Function Signature

From `flexml-compile-lite2.h`:

```c
int vaiml_compile_x2_v2(const char *config, const char *logging_level,
                        const MLADF_StringMap *input_text_files,
                        const MLADF_BinaryMap *input_binary_files,
                        MLADF_StringMap *output_text_files,
                        MLADF_BinaryMap *output_binary_files);
```

Returns `VAIML_SUCCESS (0)` on success, `VAIML_ERROR (1)` on failure.

## What Works

- Library loads without crash (unlike `libvaiml.so`)
- Function executes (doesn't segfault)
- Output directory gets `genai_config.json` and `model.onnx` files created
- Returns error code 1 (not 112/VAIML_SEGFAULT)

## What Fails

- "Could not create working dir" — the working directory path is empty string ""
- Tried all config key names: `cache_build_dir`, `build_dir`, `work_dir`, `cache_dir`, `output_dir`, `build_directory`, `work_directory`
- Tried with/without `aiecompiler_args` with `--work-dir=path` format
- Tried environment variables: `XLNX_CACHE_DIR`, `AIECOMPILER_WORK_DIR`
- Tried running from different working directories
- Tried passing ONNX model via `input_binary_files`
- All return error code 1

## Root Cause

The `cache_build_dir` config key (and all alternatives) result in an empty string being passed to `mkdir()` or `std::filesystem::create_directory()`, which returns `EINVAL` (Invalid argument). The config JSON parsing either:
1. Doesn't recognize the key for the working directory
2. Expects a different key name not yet discovered
3. Has a bug when called outside the xcompiler context

## Files

- `/media/scott/data/lfm3-npu-tool/VAIML_COMPILE_X2_V2_ATTEMPT.md` — This document
- `/tmp/vaiml_test/`, `/tmp/vaiml_test2/`, `/tmp/vaiml_env/` — Test output directories containing `genai_config.json` and `model.onnx`

## Conclusion

`vaiml_compile_x2_v2` from `libvaiml2.so` cannot be made to produce an xclbin due to the working directory creation failure. This is a fundamental limitation of the library when called outside the xcompiler tool's context.
