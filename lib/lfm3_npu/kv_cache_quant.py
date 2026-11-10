"""
KV Cache Quantization for LFM3-1.7B ONNX model.

Quantizes KV cache tensors to INT8 to reduce memory bandwidth during decode,
which is the main bottleneck for LLM inference performance.

Graph transformation per KV pair (layer L, type K/V):

  BEFORE:
    graph input:  past_key_values.L.key  (FP32)  --> GroupQueryAttention --> present.L.key (FP32) --> graph output

  AFTER:
    graph input:  past_key_values.L.key  (UINT8) --> DequantizeLinear --> past_key_values.L.key_dequantized (FP32)
                                                                     --> GroupQueryAttention --> present.L.key (FP32)
                                                                                                                    --> QuantizeLinear --> present.L.key_quantized (UINT8) --> graph output

Nodes are inserted at correct positions to maintain topological ordering.

Usage:
    python kv_cache_quant.py <input_model.onnx> <output_model.onnx>
"""

import onnx
from onnx import TensorProto, helper


def quantize_kv_cache(model_path: str, output_path: str):
    """Quantize KV cache tensors to INT8 in the ONNX model."""
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph

    # Build name -> ValueInfoProto maps for graph inputs/outputs
    input_map = {inp.name: inp for inp in graph.input}
    output_map = {out.name: out for out in graph.output}

    # Collect past_key_values graph inputs
    past_kv_inputs = {}  # (layer_idx, "key"|"value") -> ValueInfoProto
    for inp in graph.input:
        if inp.name.startswith("past_key_values."):
            parts = inp.name.split(".")
            past_kv_inputs[(parts[1], parts[2])] = inp

    # Collect present graph outputs
    present_kv_outputs = {}  # (layer_idx, "key"|"value") -> ValueInfoProto
    for out in graph.output:
        if out.name.startswith("present.") and ("key" in out.name or "value" in out.name):
            parts = out.name.split(".")
            present_kv_outputs[(parts[1], parts[2])] = out

    print(f"Found {len(past_kv_inputs)} past_key_values inputs, {len(present_kv_outputs)} present outputs")

    # Find which node consumes each past_key_values input
    consumer_idx = {}  # past_kv_input_name -> node index
    past_kv_names = {inp.name for inp in past_kv_inputs.values()}

    for idx, node in enumerate(graph.node):
        for inp_name in node.input:
            if inp_name in past_kv_names:
                consumer_idx[inp_name] = idx

    # Track where to insert DequantizeLinear (before consumer of past_key_values)
    dq_before = {}   # node index -> list of DequantizeLinear nodes

    quantized_count = 0

    for (layer_idx, kv_type), inp_tensor in sorted(
        past_kv_inputs.items(), key=lambda x: int(x[0][0])
    ):
        out_name = f"present.{layer_idx}.{kv_type}"
        out_tensor = present_kv_outputs.get((layer_idx, kv_type))
        if out_tensor is None:
            print(f"  Warning: No output found for {inp_tensor.name}, skipping")
            continue

        scale_name = f"{inp_tensor.name}_quant_scale"
        zp_name = f"{inp_tensor.name}_quant_zp"

        # Add scale and zero-point as graph initializers
        scale_tensor = helper.make_tensor(scale_name, TensorProto.FLOAT, [1], [0.007874])  # 1/127
        zp_tensor = helper.make_tensor(zp_name, TensorProto.UINT8, [1], [0])
        graph.initializer.extend([scale_tensor, zp_tensor])

        # Step 1: Change past_key_values graph input from FP32 to UINT8
        inp_tensor.type.tensor_type.elem_type = TensorProto.UINT8

        # Step 2: Add DequantizeLinear: INT8 -> FP32 (for model consumption)
        dq_name = f"{inp_tensor.name}_dequantized"
        dq_node = helper.make_node(
            "DequantizeLinear",
            inputs=[inp_tensor.name, scale_name, zp_name],
            outputs=[dq_name],
            name=f"{inp_tensor.name}_dequantize",
        )

        # Step 3: Replace past_key_values input with dequantized tensor in all node inputs
        consumer = consumer_idx.get(inp_tensor.name)
        if consumer is not None:
            dq_before.setdefault(consumer, []).append(dq_node)
        for node in graph.node:
            for i in range(len(node.input)):
                if node.input[i] == inp_tensor.name:
                    node.input[i] = dq_name

        # Step 4: Add QuantizeLinear: FP32 -> INT8 (for storage)
        q_name = f"{out_name}_quantized"
        q_node = helper.make_node(
            "QuantizeLinear",
            inputs=[out_tensor.name, scale_name, zp_name],
            outputs=[q_name],
            name=f"{out_name}_quantize",
        )

        # Insert QuantizeLinear right after the producer node
        # Find the producer node (the one that outputs out_tensor.name)
        producer_idx = None
        for idx, node in enumerate(graph.node):
            if out_tensor.name in node.output:
                producer_idx = idx
                break
        if producer_idx is not None:
            dq_before.setdefault(producer_idx + 1, []).append(q_node)
        else:
            # Append at end if no producer found
            dq_before.setdefault(len(graph.node), []).append(q_node)

        # Step 5: Update graph output: replace present.* with present.*_quantized
        # (We do this after rebuilding the node list)
        out_tensor.name = q_name
        out_tensor.type.tensor_type.elem_type = TensorProto.UINT8

        quantized_count += 1
        print(f"  Quantized {inp_tensor.name} -> {dq_name} (FP32) -> {q_name} (UINT8)")

    # Rebuild node list with DequantizeLinear inserted before consumers
    # and QuantizeLinear inserted after producers, maintaining topological order
    new_nodes = []
    for idx, node in enumerate(graph.node):
        if idx in dq_before:
            new_nodes.extend(dq_before[idx])
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    # Save model
    onnx.save(model, output_path)
    print(f"\nQuantized {quantized_count} KV cache tensors")
    print(f"Saved quantized model to: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python kv_cache_quant.py <input_model.onnx> <output_model.onnx>")
        sys.exit(1)
    quantize_kv_cache(sys.argv[1], sys.argv[2])
