#!/usr/bin/env python3
"""
Replace LFM3 Conv layers with full depthwise conv using Pad + Slice + Mul + Add (im2col style).

This replaces each depthwise Conv [1, C, L] with kernel=7, group=C, pads=3 with:
1. Pad input [1, C, L] -> [1, C, L+2*pad]
2. For each tap k: Slice padded[:, :, k:k+L] -> [1, C, L]
3. Multiply by weight tap W[:, 0, k] -> [1, C, 1] (broadcast to [1, C, L])
4. Sum all tap results

Uses dynamic shape computation (Shape + Gather + Sub + Add) to compute
correct slice end indices without knowing L statically.

All ops (Pad, Slice, Mul, Add, Shape, Gather, Sub, Add) are standard ONNX ops
supported by VitisAI EP.
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

KERNEL_SIZE = 7
PAD = 3


def convert_conv_im2col(
    input_path: str,
    output_path: str,
) -> str:
    """Replace Conv layers with full depthwise conv using Pad + Slice + Mul + Add.

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
            K = w_data.shape[2]

            print(f"  Replacing Conv {n.name}: C={C}, K={K} (full depthwise conv)")

            # === Step 1: Pad input along last dim by PAD on each side ===
            # Input: [1, C, L] -> Padded: [1, C, L+2*PAD]
            padded = uniq("pad_out")
            pads_init = add_init(
                uniq("pads"),
                np.array([0, 0, 0, 0, PAD, PAD], dtype=np.int64),
                dtype=np.int64,
            )
            pad_val_init = add_init(
                uniq("pad_val"), np.array([0.0], dtype=np.float32)
            )
            new_nodes.append(
                helper.make_node("Pad", [x_name, pads_init, pad_val_init], [padded])
            )

            # === Step 2: Compute L dynamically from padded shape ===
            # Shape(padded) -> [1, C, L+2*PAD]
            padded_shape = uniq("padded_shape")
            new_nodes.append(helper.make_node("Shape", [padded], [padded_shape]))

            # Gather index 2 -> [L+2*PAD] (scalar)
            last_dim = uniq("last_dim")
            gather_idx = add_init(
                uniq("gather_idx"),
                np.array([2], dtype=np.int64),
                dtype=np.int64,
            )
            new_nodes.append(
                helper.make_node("Gather", [padded_shape, gather_idx], [last_dim])
            )

            # L = L+2*PAD - 2*PAD
            L_dim = uniq("L_dim")
            two_pad = add_init(
                uniq("two_pad"),
                np.array([2 * PAD], dtype=np.int64),
                dtype=np.int64,
            )
            new_nodes.append(
                helper.make_node("Sub", [last_dim, two_pad], [L_dim])
            )

            # === Step 3: For each tap, Slice padded[:, :, k:k+L] and Multiply ===
            tap_results = []
            for tap_k in range(K):
                # end_k = L + k
                end_k = uniq(f"end_k_{tap_k}")
                k_init = add_init(
                    uniq(f"tap_k_{tap_k}"),
                    np.array([tap_k], dtype=np.int64),
                    dtype=np.int64,
                )
                new_nodes.append(
                    helper.make_node("Add", [L_dim, k_init], [end_k])
                )

                # Slice padded[:, :, k:end_k] -> [1, C, L]
                slice_out = uniq(f"slice_{tap_k}")
                starts = add_init(
                    uniq(f"starts_{tap_k}"),
                    np.array([0, 0, tap_k], dtype=np.int64),
                    dtype=np.int64,
                )

                # Build ends = [1, C, end_k] by concatenating [1], [C], [end_k]
                # All must be rank-3 tensors for Concat
                ones_1_val = add_init(
                    uniq(f"ones_1_val_{tap_k}"),
                    np.array([1], dtype=np.int64),
                    dtype=np.int64,
                )
                ones_C_val = add_init(
                    uniq(f"ones_C_val_{tap_k}"),
                    np.array([C], dtype=np.int64),
                    dtype=np.int64,
                )

                # Reshape [1] -> [1,1,1]
                ones_1_r = uniq(f"ones_1_r_{tap_k}")
                ones_1_shape = add_init(
                    uniq(f"ones_1_shape_{tap_k}"),
                    np.array([1, 1, 1], dtype=np.int64),
                    dtype=np.int64,
                )
                new_nodes.append(
                    helper.make_node("Reshape", [ones_1_val, ones_1_shape], [ones_1_r])
                )

                # Reshape [C] -> [1,1,1]
                ones_C_r = uniq(f"ones_C_r_{tap_k}")
                ones_C_shape = add_init(
                    uniq(f"ones_C_shape_{tap_k}"),
                    np.array([1, 1, 1], dtype=np.int64),
                    dtype=np.int64,
                )
                new_nodes.append(
                    helper.make_node("Reshape", [ones_C_val, ones_C_shape], [ones_C_r])
                )

                # Reshape end_k from scalar to [1,1,1]
                end_k_reshaped = uniq(f"end_k_reshaped_{tap_k}")
                end_k_shape = add_init(
                    uniq(f"end_k_shape_{tap_k}"),
                    np.array([1, 1, 1], dtype=np.int64),
                    dtype=np.int64,
                )
                new_nodes.append(
                    helper.make_node("Reshape", [end_k, end_k_shape], [end_k_reshaped])
                )

                # Concat [1, C, end_k] -> [1, C, end_k] (3D)
                ends_concat = uniq(f"ends_concat_{tap_k}")
                new_nodes.append(
                    helper.make_node(
                        "Concat",
                        [ones_1_r, ones_C_r, end_k_reshaped],
                        [ends_concat],
                        axis=0,
                    )
                )

                new_nodes.append(
                    helper.make_node(
                        "Slice", [padded, starts, ends_concat], [slice_out]
                    )
                )

                # Weight tap: W[:, 0, tap_k] -> [1, C, 1]
                w_tap_data = w_data[:, 0, tap_k].reshape(1, C, 1)
                w_tap_name = add_init(uniq(f"w_tap_{tap_k}"), w_tap_data)

                # Mul: slice_k * w_tap -> [1, C, L]
                tap_mul = uniq(f"tap_mul_{tap_k}")
                new_nodes.append(
                    helper.make_node("Mul", [slice_out, w_tap_name], [tap_mul])
                )
                tap_results.append(tap_mul)

            # === Step 3: Sum all tap results ===
            acc = tap_results[0]
            for i in range(1, len(tap_results)):
                new_acc = uniq(f"acc_{i}")
                new_nodes.append(
                    helper.make_node("Add", [acc, tap_results[i]], [new_acc])
                )
                acc = new_acc

            # Final output
            new_nodes.append(helper.make_node("Identity", [acc], [out_name]))

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

    past_conv_names = {
        i.name for i in model.graph.input if i.name.startswith("past_conv.")
    }
    new_inputs = [
        inp for inp in model.graph.input if inp.name not in past_conv_names
    ]

    present_conv_names = {
        o.name for o in model.graph.output if o.name.startswith("present_conv.")
    }
    new_outputs = [
        out for out in model.graph.output if out.name not in present_conv_names
    ]

    print(
        f"Removed {len(past_conv_names)} past_conv inputs, "
        f"{len(present_conv_names)} present_conv outputs"
    )
    print(f"Total nodes: {len(new_nodes)}")

    graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        new_inputs,
        new_outputs,
        new_initializers,
    )

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


def convert_centertap(
    input_path: str,
    output_path: str,
) -> str:
    """Replace Conv layers with center-tap weight multiplication (original method)."""
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
            w_data = numpy_helper.to_array(w_init)
            C = w_data.shape[0]
            center_tap = KERNEL_SIZE // 2

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
    """Replace Conv layers with Identity (complete bypass)."""
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

    graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        new_inputs,
        new_outputs,
        list(model.graph.initializer),
    )
    new_model = helper.make_model(graph, opset_imports=model.opset_import)

    onnx.save_model(
        new_model,
        output_path,
        save_as_external_data=False,
        size_threshold=2**63 - 1,
    )
    print(f"\nSaved: {output_path}")
    return output_path