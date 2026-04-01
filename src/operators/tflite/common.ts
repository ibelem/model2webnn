// Shared helpers for TFLite operator builders

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';

/**
 * Emit fused activation after an op.
 * TFLite convolutions, pools, and element-wise ops can have fused activations.
 */
export function emitFusedActivation(varName: string, activation: string | undefined, emitter: CodeEmitter): string {
  if (!activation || activation === 'NONE') return varName;

  const activatedVar = `${varName}_activated`;
  switch (activation) {
    case 'RELU':
      emitter.line(`const ${activatedVar} = builder.relu(${varName});`);
      return activatedVar;
    case 'RELU6':
      emitter.line(`const ${activatedVar} = builder.clamp(${varName}, { minValue: 0, maxValue: 6 });`);
      return activatedVar;
    case 'RELU_N1_TO_1':
      emitter.line(`const ${activatedVar} = builder.clamp(${varName}, { minValue: -1, maxValue: 1 });`);
      return activatedVar;
    case 'TANH':
      emitter.line(`const ${activatedVar} = builder.tanh(${varName});`);
      return activatedVar;
    default:
      return varName;
  }
}

/**
 * Compute explicit SAME padding for NHWC tensors.
 * TFLite SAME: output = ceil(input / stride)
 * Extra padding goes to bottom/right (same-upper convention).
 * WebNN does not support autoPad — all padding must be explicit.
 */
export function computeSamePadding(
  inputH: number, inputW: number,
  kernelH: number, kernelW: number,
  strideH: number, strideW: number,
  dilationH: number, dilationW: number
): [number, number, number, number] {
  const effectiveKH = (kernelH - 1) * dilationH + 1;
  const effectiveKW = (kernelW - 1) * dilationW + 1;
  const outputH = Math.ceil(inputH / strideH);
  const outputW = Math.ceil(inputW / strideW);
  const totalPadH = Math.max(0, (outputH - 1) * strideH + effectiveKH - inputH);
  const totalPadW = Math.max(0, (outputW - 1) * strideW + effectiveKW - inputW);
  const padTop = Math.floor(totalPadH / 2);
  const padBottom = totalPadH - padTop;
  const padLeft = Math.floor(totalPadW / 2);
  const padRight = totalPadW - padLeft;
  return [padTop, padBottom, padLeft, padRight];
}

/** Map WebNN data types to typed array names */
export const dataTypeToArray: Record<string, string> = {
  float32: 'Float32Array', float16: 'Float16Array',
  int32: 'Int32Array', uint32: 'Uint32Array',
  int8: 'Int8Array', uint8: 'Uint8Array',
  int64: 'BigInt64Array', uint64: 'BigUint64Array',
};

/**
 * Check if a tensor needs dequantization (int8/uint8) and emit dequantizeLinear.
 * Returns the (possibly new) variable name to use for the dequantized tensor.
 * If the tensor is already float, returns the original variable unchanged.
 *
 * @param varName - current JS variable name for the operand
 * @param tensorName - original tensor name (for type lookup)
 * @param localInputIdx - position in node.inputs (0, 1, 2) for quantization param lookup
 * @param node - the operator node (attributes contain quantization params)
 * @param emitter - code emitter
 * @param prefix - prefix for generated variable names
 */
export function emitDequantizeIfNeeded(
  varName: string,
  tensorName: string,
  localInputIdx: number,
  node: NodeIR,
  emitter: CodeEmitter,
  prefix: string,
): string {
  const dtype = emitter.tensorDataType(tensorName);
  if (!dtype || (dtype !== 'int8' && dtype !== 'uint8' && dtype !== 'int32' && dtype !== 'uint32')) return varName;

  const scaleValues = node.attributes[`input_${localInputIdx}_scale`] as number[] | undefined;
  const zpValues = node.attributes[`input_${localInputIdx}_zero_point`] as number[] | undefined;

  if (!scaleValues || scaleValues.length === 0) {
    // No quantization params — fall back to cast
    emitter.comment(`Cast ${dtype} → float32 (no quantization params)`);
    const castVar = `${prefix}_f32`;
    emitter.line(`const ${castVar} = builder.cast(${varName}, 'float32');`);
    return castVar;
  }

  // Decompose dequantization into cast + sub + mul.
  // Formula: output = (cast(input, float32) - cast(zeroPoint, float32)) * scale
  //
  // For per-channel quantization the scale/zp vector must be reshaped so its
  // single non-trivial dim aligns with quantized_dimension, otherwise NumPy-
  // style broadcasting (from the right) will mismatch.
  emitter.comment(`Dequantize ${dtype} → float32`);
  const castVar = `${prefix}_f32`;
  const scaleVar = `${prefix}_scale`;
  const dqVar = `${prefix}_dq`;

  emitter.line(`const ${castVar} = builder.cast(${varName}, 'float32');`);

  // Determine the shape for per-channel scale/zp constants.
  // For per-tensor (length 1) use scalar []; for per-channel, reshape to match
  // the quantized dimension so broadcasting works with any tensor rank.
  const quantizedDim = node.attributes[`input_${localInputIdx}_quantized_dimension`] as number | undefined;
  const tensorShape = emitter.tensorShape(tensorName);
  const tensorRank = tensorShape ? tensorShape.length : 0;

  function broadcastShape(len: number): string {
    if (len === 1) return '[]';
    // Per-channel: if we know the tensor rank and quantized dim, reshape to
    // e.g. [32, 1, 1, 1] so it broadcasts with [32, 3, 3, 12].
    if (quantizedDim != null && tensorRank > 1) {
      const shape = new Array(tensorRank).fill(1);
      shape[quantizedDim] = len;
      return `[${shape.join(', ')}]`;
    }
    // Fallback: 1-D — works when quantized dim is the last dim (default broadcasting)
    return `[${len}]`;
  }

  const scaleShape = broadcastShape(scaleValues.length);
  emitter.line(`const ${scaleVar} = builder.constant({dataType: 'float32', shape: ${scaleShape}}, new Float32Array([${scaleValues.join(', ')}]));`);

  // Zero-point subtraction (skip if all zeros)
  const zpArr = zpValues ?? [0];
  const allZerosZp = zpArr.every((v) => v === 0);

  if (allZerosZp) {
    emitter.line(`const ${dqVar} = builder.mul(${castVar}, ${scaleVar});`);
  } else {
    const zpConstVar = `${prefix}_zpf`;
    const zpShape = broadcastShape(zpArr.length);
    emitter.line(`const ${zpConstVar} = builder.constant({dataType: 'float32', shape: ${zpShape}}, new Float32Array([${zpArr.map((v) => v.toString()).join(', ')}]));`);
    emitter.line(`const ${dqVar} = builder.mul(builder.sub(${castVar}, ${zpConstVar}), ${scaleVar});`);
  }

  return dqVar;
}
