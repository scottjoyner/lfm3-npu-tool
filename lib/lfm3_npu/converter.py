"""
Model conversion utilities for LFM3-1.7B NPU compatibility.

Replaces Conv layers with simplified ONNX ops that VitisAI EP supports,
avoiding the libvaiml.so::vaiml_get_capability_v5 crash.
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

from .replace_conv_im2col import convert_conv_im2col

KERNEL_SIZE = 7
PAD = 3


def convert_centertap(
    input_path: str,
    output_path: str,
) -> str:
    """Replace Conv layers with center-tap weight multiplication.

    This replaces each depthwise Conv with:
        output[c, t] = input[c, t] * w[c, 0, 3]

    Uses standard ONNX Slice + Reshape + Mul ops that VitisAI EP supports.
    Preserves the center weight of the depthwise convolution for better
    quality than full Identity bypass.

    Args:
        input_path: Path to the original ONNX model.
        output_path: Path to save the converted model.

    Returns:
        Path to the saved model file.
    """
    model = onnx.load(input_path, load_external_data=True)
    init_map = {init.name: init for init in model.graph.initializer}

    conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
    concat_nodes = [
        n
        for n in model.graph.node
        if n.op_type == "Concat"
        and any(i.startswith("past_conv.") for i in n.input)
    ]

    print(f"Found {len(conv_nodes)} Conv nodes, {len(concat_nodes)} Conv_Input Concat nodes")

    new_nodes = []
    new_initializers = list(model.graph.initializer)
    counter = [0]

    def uniq(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_{counter[0]}"

    def add_init(name: str, array: np.ndarray, dtype: np.dtype = np.float32):
        init = numpy_helper.from_array(np.asarray(array, dtype=dtype), name=name)
        new_initializers.append(init)
        return name

    concat_output_replacement: dict = {}

    for concat_node in concat_nodes:
        non_past_inputs = [
            i for i in concat_node.input if not i.startswith("past_conv.")
        ]
        if non_past_inputs:
            concat_output_replacement[concat_node.output[0]] = non_past_inputs[0]

    for n in model.graph.node:
        if n.op_type == "Conv":
            x_name = n.input[0]
            w_name = n.input[1]
            out_name = n.output[0]

            if x_name in concat_output_replacement:
                x_name = concat_output_replacement[x_name]

            w_init = init_map[w_name]
            w_data = numpy_helper.to_array(w_init)  # [C, 1, K]
            C = w_data.shape[0]
            center_tap = KERNEL_SIZE // 2  # 3

            print(f"  Replacing Conv {n.name}: C={C}, K={KERNEL_SIZE}, center_tap={center_tap}")

            # Extract center tap weight: W[:, 0, center_tap] -> [C]
            center_w_slice = uniq("center_w_slice")
            w_starts = add_init(
                uniq("w_starts"),
                np.array([0, 0, center_tap], dtype=np.int64),
                dtype=np.int64,
            )
            w_ends = add_init(
                uniq("w_ends"),
                np.array([C, 1, center_tap + 1], dtype=np.int64),
                dtype=np.int64,
            )
            new_nodes.append(
                helper.make_node("Slice", [w_name, w_starts, w_ends], [center_w_slice])
            )

            # Reshape [C, 1, 1] -> [1, C, 1] for broadcasting
            center_w_reshaped = uniq("center_w_reshaped")
            w_tap_shape = add_init(
                uniq("w_tap_shape"),
                np.array([1, C, 1], dtype=np.int64),
                dtype=np.int64,
            )
            new_nodes.append(
                helper.make_node(
                    "Reshape", [center_w_slice, w_tap_shape], [center_w_reshaped]
                )
            )

            # Mul: input * center_weight -> output
            new_nodes.append(
                helper.make_node("Mul", [x_name, center_w_reshaped], [out_name])
            )

        elif n.op_type == "Concat" and any(
            i.startswith("past_conv.") for i in n.input
        ):
            non_past_inputs = [i for i in n.input if not i.startswith("past_conv.")]
            if non_past_inputs:
                new_nodes.append(
                    helper.make_node(
                        "Identity", non_past_inputs, list(n.output), name=n.name
                    )
                )
            else:
                new_nodes.append(n)
        else:
            new_nodes.append(n)

    past_conv_names = {i.name for i in model.graph.input if i.name.startswith("past_conv.")}
    new_inputs = [inp for inp in model.graph.input if inp.name not in past_conv_names]

    present_conv_names = {o.name for o in model.graph.output if o.name.startswith("present_conv.")}
    new_outputs = [out for out in model.graph.output if out.name not in present_conv_names]

    graph = helper.make_graph(new_nodes, model.graph.name, new_inputs, new_outputs, new_initializers)
    new_model = helper.make_model(graph, opset_imports=model.opset_import)

    onnx.save_model(
        new_model,
        output_path,
        save_as_external_data=False,
        size_threshold=2**63 - 1,
    )
    print(f"\nSaved: {output_path}")
    return output_path


def convert_bypass(
    input_path: str,
    output_path: str,
) -> str:
    """Replace Conv layers with Identity (complete bypass).

    This removes all Conv computation entirely. Fastest but lowest quality
    as no local context mixing occurs.

    Args:
        input_path: Path to the original ONNX model.
        output_path: Path to save the converted model.

    Returns:
        Path to the saved model file.
    """
    model = onnx.load(input_path, load_external_data=True)

    conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
    concat_nodes = [
        n
        for n in model.graph.node
        if n.op_type == "Concat"
        and any(i.startswith("past_conv.") for i in n.input)
    ]

    print(f"Found {len(conv_nodes)} Conv nodes, {len(concat_nodes)} Conv_Input Concat nodes")

    new_nodes = []
    for n in model.graph.node:
        if n.op_type == "Conv":
            x_name = n.input[0]
            out_name = n.output[0]
            new_nodes.append(
                helper.make_node("Identity", [x_name], [out_name], name=n.name)
            )
        elif n.op_type == "Concat" and any(
            i.startswith("past_conv.") for i in n.input
        ):
            non_past_inputs = [i for i in n.input if not i.startswith("past_conv.")]
            if non_past_inputs:
                new_nodes.append(
                    helper.make_node(
                        "Identity", non_past_inputs, list(n.output), name=n.name
                    )
                )
            else:
                new_nodes.append(n)
        else:
            new_nodes.append(n)

    past_conv_names = {i.name for i in model.graph.input if i.name.startswith("past_conv.")}
    new_inputs = [inp for inp in model.graph.input if inp.name not in past_conv_names]

    present_conv_names = {o.name for o in model.graph.output if o.name.startswith("present_conv.")}
    new_outputs = [out for out in model.graph.output if out.name not in present_conv_names]

    graph = helper.make_graph(new_nodes, model.graph.name, new_inputs, new_outputs, list(model.graph.initializer))
    new_model = helper.make_model(graph, opset_imports=model.opset_import)

    onnx.save_model(
        new_model,
        output_path,
        save_as_external_data=False,
        size_threshold=2**63 - 1,
    )
    print(f"\nSaved: {output_path}")
    return output_path


def convert(
    input_path: str,
    output_path: str,
    method: str = "centertap",
) -> str:
    """Convert LFM3 ONNX model to NPU-compatible format.

    Args:
        input_path: Path to the original ONNX model.
        output_path: Path to save the converted model.
        method: Conversion method - 'im2col' (full depthwise conv),
                'centertap' (center tap only), or 'bypass' (identity only).

    Returns:
        Path to the saved model file.
    """
    if method == "im2col":
        return convert_conv_im2col(input_path, output_path)
    elif method == "centertap":
        return convert_centertap(input_path, output_path)
    elif method == "bypass":
        return convert_bypass(input_path, output_path)
    else:
        raise ValueError(
            f"Unknown conversion method: {method}. "
            "Use 'im2col', 'centertap', or 'bypass'."
        )


# Alias for backward compatibility
convert_model = convert