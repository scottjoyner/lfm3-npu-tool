"""Offline CPU inference contract using a tiny synthetic ONNX graph.

This tests ONNX Runtime CPUExecutionProvider and the project's greedy-token
selection helper. It does NOT validate LFM3 model weights, text generation,
systemd services, network endpoints, or NPU hardware.
"""

from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lfm3_npu.inference import _sample_token


def _cpu_session():
    """Create a deterministic, untrained 2->3 linear-logits ONNX fixture."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])
    weight = helper.make_tensor(
        "weight", TensorProto.FLOAT, [2, 3], [2.0, 0.0, 1.0, 1.0, -1.0, 0.0]
    )
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [3], [0.5, 0.25, -0.25])
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "weight"], ["linear"]),
            helper.make_node("Add", ["linear", "bias"], ["logits"]),
        ],
        "synthetic_cpu_logits",
        [x],
        [logits],
        initializer=[weight, bias],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="lfm3-ci"
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    assert session.get_providers() == ["CPUExecutionProvider"]
    return session


@pytest.mark.parametrize(
    ("input_values", "expected_logits", "expected_token"),
    [
        ([2.0, 1.0], [5.5, -0.75, 1.75], 0),
        ([1.0, -3.0], [-0.5, 3.25, 0.75], 1),
    ],
)
def test_cpu_runtime_logits_and_lfm_greedy_selection(
    input_values, expected_logits, expected_token
):
    session = _cpu_session()
    results = session.run(
        ["logits"], {"X": np.asarray([input_values], dtype=np.float32)}
    )
    assert len(results) == 1
    actual = results[0]
    assert actual.shape == (1, 3)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual[0], np.asarray(expected_logits, dtype=np.float32), rtol=0, atol=1e-6
    )
    assert _sample_token(actual[0], temperature=0.0, top_p=1.0) == expected_token


def test_cpu_runtime_rejects_wrong_tensor_type():
    session = _cpu_session()
    with pytest.raises(Exception, match="Unexpected input data type"):
        session.run(["logits"], {"X": np.asarray([[2, 1]], dtype=np.float64)})
