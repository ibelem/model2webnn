// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/matMulNBits_op_builder.cc
// ONNX MatMulNBits → WebNN decomposition: dequantize + transpose + matmul
// Supports 4-bit quantized weight matrices.

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

function emitMatMulNBits(node: NodeIR, emitter: CodeEmitter): void {
  const inputA = emitter.ref(node.inputs[0]);  // [M, K] float
  const b = emitter.ref(node.inputs[1]);        // quantized weights [N, n_blocks, blob_size]
  const scales = emitter.ref(node.inputs[2]);    // [N, n_blocks]
  const output = emitter.declare(node.outputs[0]);

  const K = (node.attributes.K as number) ?? 0;
  const N = (node.attributes.N as number) ?? 0;
  const bits = (node.attributes.bits as number) ?? 4;

  emitter.comment(`MatMulNBits — K=${K}, N=${N}, bits=${bits}`);

  if (bits !== 4) {
    emitter.comment('Only 4-bit quantization is supported');
    emitter.line(`const ${output} = builder.matmul(${inputA}, ${b}); // UNSUPPORTED: ${bits}-bit`);
    return;
  }

  // Reshape scales for broadcasting: [N, n_blocks] → [N, n_blocks, 1]
  const scalesReshaped = `${output}_scales_r`;
  emitter.line(`const ${scalesReshaped} = builder.reshape(${scales}, [0, 0, 1]);`);

  // Zero points
  const hasZeroPoints = node.inputs.length > 3 && node.inputs[3] !== '';
  let zpVar: string;
  if (hasZeroPoints) {
    zpVar = emitter.ref(node.inputs[3]);
  } else {
    // Default zero point = 8 for 4-bit
    zpVar = `${output}_default_zp`;
    emitter.line(`const ${zpVar} = builder.constant({dataType: 'uint8', shape: []}, new Uint8Array([8]));`);
  }

  // Dequantize: B_float = dequantize(B_uint4, scales, zero_points) → [N, K]
  const dequantized = `${output}_dq`;
  emitter.line(`const ${dequantized} = builder.dequantizeLinear(${b}, { scale: ${scalesReshaped}, zeroPoint: ${zpVar} });`);

  // Reshape dequantized to [N, K]
  const dqReshaped = `${output}_dq_r`;
  emitter.line(`const ${dqReshaped} = builder.reshape(${dequantized}, [${N}, ${K}]);`);

  // Transpose to [K, N]
  const dqTransposed = `${output}_dq_t`;
  emitter.line(`const ${dqTransposed} = builder.transpose(${dqReshaped}, { permutation: [1, 0] });`);

  // MatMul: [M, K] × [K, N] = [M, N]
  const matmulResult = `${output}_mm`;
  emitter.line(`const ${matmulResult} = builder.matmul(${inputA}, ${dqTransposed});`);

  // Add bias if present
  const hasBias = node.inputs.length > 5 && node.inputs[5] !== '';
  if (hasBias) {
    const bias = emitter.ref(node.inputs[5]);
    emitter.line(`const ${output} = builder.add(${matmulResult}, ${bias});`);
  } else {
    emitter.line(`const ${output} = ${matmulResult};`);
  }
}

registerOnnxOp('MatMulNBits', emitMatMulNBits);
