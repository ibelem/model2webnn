// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/matMulNBits_op_builder.cc
// ONNX MatMulNBits → WebNN decomposition: dequantize + transpose + matmul
// Supports 4-bit quantized weight matrices.
//
// Before this emitter runs, fixMatMulNBitsConstants() (in index.ts) has already
// reinterpreted B from uint8 [N, n_blocks, blob_size] to uint4 [N, n_blocks, blob_size*2]
// and zero_points from uint8 to uint4 [N, n_blocks, 1].  Raw bytes are unchanged — WebNN's
// uint4 type interprets each byte as two packed nibbles.

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

function emitMatMulNBits(node: NodeIR, emitter: CodeEmitter): void {
  const inputA = emitter.ref(node.inputs[0]);  // [M, K] float
  const b = emitter.ref(node.inputs[1]);        // uint4 weights [N, n_blocks, block_size]
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
  const blockSize = (node.attributes.block_size as number) ?? 32;
  const nBlocks = Math.ceil(K / blockSize);
  const scalesShape = emitter.tensorShape(node.inputs[2]);
  const scalesReshaped = `${output}_scales_r`;
  if (scalesShape && scalesShape.length === 2 && scalesShape.every((d): d is number => typeof d === 'number')) {
    emitter.line(`const ${scalesReshaped} = builder.reshape(${scales}, [${scalesShape[0]}, ${scalesShape[1]}, 1]);`);
  } else {
    emitter.line(`const ${scalesReshaped} = builder.reshape(${scales}, [${N}, ${nBlocks}, 1]);`);
  }

  // Zero points — must be uint4 with shape [N, n_blocks, 1] (same as reshaped scales)
  // See ORT matMulNBits_op_builder.cc: x_zero_point uses x_scale_shape_array
  const hasZeroPoints = node.inputs.length > 3 && node.inputs[3] !== '';
  let zpVar: string;
  if (hasZeroPoints) {
    // Zero points constant was already reinterpreted as uint4 [N, n_blocks, 1]
    // by fixMatMulNBitsConstants — use directly, no reshape needed.
    zpVar = emitter.ref(node.inputs[3]);
  } else {
    // Default zero point = 8 for unsigned 4-bit (midpoint: 2³).
    // ORT packs 8 into both nibbles of each byte: (8 | (8 << 4)) = 0x88 = 136.
    const numUint4Elements = N * nBlocks;
    const numBytes = Math.ceil(numUint4Elements / 2);
    zpVar = `${output}_default_zp`;
    emitter.line(`const ${zpVar} = builder.constant({dataType: 'uint4', shape: [${N}, ${nBlocks}, 1]}, new Uint8Array(${numBytes}).fill(136));`);
  }

  // DequantizeLinear: B_float = dequantize(B_uint4, scales, zero_points)
  // B is uint4 [N, n_blocks, block_size], output is float [N, n_blocks, block_size]
  const dequantized = `${output}_dq`;
  emitter.line(`const ${dequantized} = builder.dequantizeLinear(${b}, ${scalesReshaped}, ${zpVar});`);

  // Reshape to [N, K]
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
