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
  // Use actual dimensions (WebNN reshape does not support ONNX's "0 = copy from input")
  const blockSize = (node.attributes.block_size as number) ?? 32;
  const nBlocks = Math.ceil(K / blockSize);
  const scalesShape = emitter.tensorShape(node.inputs[2]);
  const scalesReshaped = `${output}_scales_r`;
  if (scalesShape && scalesShape.length === 2 && scalesShape.every((d): d is number => typeof d === 'number')) {
    emitter.line(`const ${scalesReshaped} = builder.reshape(${scales}, [${scalesShape[0]}, ${scalesShape[1]}, 1]);`);
  } else {
    // Fallback: use N and compute n_blocks from K
    emitter.line(`const ${scalesReshaped} = builder.reshape(${scales}, [${N}, ${nBlocks}, 1]);`);
  }

  // Zero points — must have the same shape as reshaped scales [N, n_blocks_per_col, 1]
  // See ORT matMulNBits_op_builder.cc: x_zero_point uses x_scale_shape_array
  const hasZeroPoints = node.inputs.length > 3 && node.inputs[3] !== '';
  let zpVar: string;
  if (hasZeroPoints) {
    // Reshape provided zero points to match scale shape [N, n_blocks, 1]
    const zpRaw = emitter.ref(node.inputs[3]);
    zpVar = `${output}_zp_r`;
    emitter.line(`const ${zpVar} = builder.reshape(${zpRaw}, [${N}, ${nBlocks}, 1]);`);
  } else {
    // Default zero point = 8 for 4-bit, with shape matching reshaped scales
    const numElements = N * nBlocks * 1;
    zpVar = `${output}_default_zp`;
    emitter.line(`const ${zpVar} = builder.constant({dataType: 'uint8', shape: [${N}, ${nBlocks}, 1]}, new Uint8Array(${numElements}).fill(8));`);
  }

  // Dequantize: B_float = dequantize(B_uint4, scales, zero_points) → [N, K]
  // WebNN signature: dequantizeLinear(input, scale, zeroPoint)
  const dequantized = `${output}_dq`;
  emitter.line(`const ${dequantized} = builder.dequantizeLinear(${b}, ${scalesReshaped}, ${zpVar});`);

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
