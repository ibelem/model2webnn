// TFLite miscellaneous ops: CAST, ARG_MAX, ARG_MIN, DEPTH_TO_SPACE, SPACE_TO_DEPTH,
// CUMSUM, DEQUANTIZE, QUANTIZE, BROADCAST_TO, WHERE, SELECT_V2, SELECT,
// ZEROS_LIKE, ADD_N, IDENTITY

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';

function emitCast(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const outType = (node.attributes.out_data_type as string) ?? 'FLOAT32';
  emitter.line(`const ${output} = builder.cast(${input}, '${outType.toLowerCase()}');`);
}

function emitArgMax(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // TFLite ARG_MAX: inputs[1] is the axis (constant int32 scalar)
  // WebNN argMax(input, axis, options?) — axis is required unsigned long
  let axis = 0;
  if (node.inputs.length > 1) {
    const axisValues = emitter.constantIntValues(node.inputs[1]);
    if (axisValues) {
      axis = axisValues[0];
      if (axis < 0) {
        const inputShape = emitter.tensorShape(node.inputs[0]);
        if (inputShape) axis += inputShape.length;
      }
    }
  }
  const keepDims = (node.attributes.keep_dims as boolean) ?? true;
  emitter.line(`const ${output} = builder.argMax(${input}, ${axis}${keepDims ? '' : ', { keepDimensions: false }'});`);
}

function emitArgMin(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // TFLite ARG_MIN: inputs[1] is the axis (constant int32 scalar)
  let axis = 0;
  if (node.inputs.length > 1) {
    const axisValues = emitter.constantIntValues(node.inputs[1]);
    if (axisValues) {
      axis = axisValues[0];
      if (axis < 0) {
        const inputShape = emitter.tensorShape(node.inputs[0]);
        if (inputShape) axis += inputShape.length;
      }
    }
  }
  const keepDims = (node.attributes.keep_dims as boolean) ?? true;
  emitter.line(`const ${output} = builder.argMin(${input}, ${axis}${keepDims ? '' : ', { keepDimensions: false }'});`);
}

// WebNN does not have depthToSpace; decompose into reshape → transpose → reshape.
// TFLite uses NHWC layout.
function emitDepthToSpace(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const blockSize = (node.attributes.block_size as number) ?? 2;

  const inputShape = emitter.tensorShape(node.inputs[0]);
  if (!inputShape || inputShape.length !== 4) {
    emitter.comment(`DepthToSpace: unknown input shape, cannot decompose`);
    return;
  }
  // NHWC: [batch, height, width, channels]
  const [batch, height, width, channels] = inputShape;
  const newC = `${Number(channels) / (blockSize * blockSize)}`;
  const newH = `${Number(height) * blockSize}`;
  const newW = `${Number(width) * blockSize}`;

  // Reshape [b, h, w, c] → [b, h, w, bs, bs, newC]
  const shape1 = `[${batch}, ${height}, ${width}, ${blockSize}, ${blockSize}, ${newC}]`;
  // Transpose → [b, h, bs, w, bs, newC]
  const perm = `[0, 1, 3, 2, 4, 5]`;
  // Reshape → [b, newH, newW, newC]
  const shape2 = `[${batch}, ${newH}, ${newW}, ${newC}]`;

  emitter.line(`const ${output}_r1 = builder.reshape(${input}, ${shape1});`);
  emitter.line(`const ${output}_t = builder.transpose(${output}_r1, { permutation: ${perm} });`);
  emitter.line(`const ${output} = builder.reshape(${output}_t, ${shape2});`);
}

// WebNN does not have spaceToDepth; decompose into reshape → transpose → reshape.
// TFLite uses NHWC layout.
function emitSpaceToDepth(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const blockSize = (node.attributes.block_size as number) ?? 2;

  const inputShape = emitter.tensorShape(node.inputs[0]);
  if (!inputShape || inputShape.length !== 4) {
    emitter.comment(`SpaceToDepth: unknown input shape, cannot decompose`);
    return;
  }
  // NHWC: [batch, height, width, channels]
  const [batch, height, width, channels] = inputShape;
  const newC = `${Number(channels) * blockSize * blockSize}`;
  const newH = `${Number(height) / blockSize}`;
  const newW = `${Number(width) / blockSize}`;

  // Reshape [b, h, w, c] → [b, h/bs, bs, w/bs, bs, c]
  const shape1 = `[${batch}, ${newH}, ${blockSize}, ${newW}, ${blockSize}, ${channels}]`;
  // Transpose → [b, h/bs, w/bs, bs, bs, c]
  const perm = `[0, 1, 3, 2, 4, 5]`;
  // Reshape → [b, h/bs, w/bs, c*bs*bs]
  const shape2 = `[${batch}, ${newH}, ${newW}, ${newC}]`;

  emitter.line(`const ${output}_r1 = builder.reshape(${input}, ${shape1});`);
  emitter.line(`const ${output}_t = builder.transpose(${output}_r1, { permutation: ${perm} });`);
  emitter.line(`const ${output} = builder.reshape(${output}_t, ${shape2});`);
}

function emitCumSum(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // TFLite CUMSUM: inputs[1] is the axis (constant int32 scalar)
  let axis = 0;
  if (node.inputs.length > 1) {
    const axisValues = emitter.constantIntValues(node.inputs[1]);
    if (axisValues) {
      axis = axisValues[0];
      if (axis < 0) {
        const inputShape = emitter.tensorShape(node.inputs[0]);
        if (inputShape) axis += inputShape.length;
      }
    }
  }
  emitter.line(`const ${output} = builder.cumulativeSum(${input}, ${axis});`);
}

// TFLite DEQUANTIZE — quantization params are in tensor metadata (stored as attributes by parser)
// WebNN signature: builder.dequantizeLinear(input, scale, zeroPoint)
// Spec constraint: scale and zeroPoint must have the same rank as input.
function emitDequantize(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.comment('DEQUANTIZE');

  const scaleValues = findQuantizationScale(node);
  const zpValues = findQuantizationZeroPoint(node);

  if (scaleValues && scaleValues.length > 0) {
    // Decompose: output = (cast(input, float32) - zeroPoint) * scale
    // Using cast + sub + mul avoids dequantizeLinear's rank constraints.
    const castVar = `${output}_f32`;
    const scaleVar = `${output}_scale`;
    emitter.line(`const ${castVar} = builder.cast(${input}, 'float32');`);

    const scaleShape = scaleValues.length === 1 ? '[]' : `[${scaleValues.length}]`;
    emitter.line(`const ${scaleVar} = builder.constant({dataType: 'float32', shape: ${scaleShape}}, new Float32Array([${scaleValues.join(', ')}]));`);

    const zpArr = zpValues ?? [0];
    const allZerosZp = zpArr.every((v) => v === 0);

    if (allZerosZp) {
      emitter.line(`const ${output} = builder.mul(${castVar}, ${scaleVar});`);
    } else {
      const zpConstVar = `${output}_zpf`;
      const zpShape = zpArr.length === 1 ? '[]' : `[${zpArr.length}]`;
      emitter.line(`const ${zpConstVar} = builder.constant({dataType: 'float32', shape: ${zpShape}}, new Float32Array([${zpArr.map((v) => v.toString()).join(', ')}]));`);
      emitter.line(`const ${output} = builder.mul(builder.sub(${castVar}, ${zpConstVar}), ${scaleVar});`);
    }
  } else {
    emitter.comment('No quantization params found — using cast as fallback');
    emitter.line(`const ${output} = builder.cast(${input}, 'float32');`);
  }
}

// TFLite QUANTIZE — quantization params are in output tensor metadata
// Spec constraint: scale and zeroPoint must have the same rank as input.
function emitQuantize(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.comment('QUANTIZE');

  const scaleValues = findQuantizationScale(node);
  const zpValues = findQuantizationZeroPoint(node);

  if (scaleValues && scaleValues.length > 0) {
    // Decompose: output = cast(round(input / scale + zeroPoint), outputDtype)
    // Using cast + div + add avoids quantizeLinear's rank constraints.
    const outputDtype = emitter.tensorDataType(node.outputs[0]) ?? 'uint8';
    const scaleVar = `${output}_scale`;
    const scaleShape = scaleValues.length === 1 ? '[]' : `[${scaleValues.length}]`;
    emitter.line(`const ${scaleVar} = builder.constant({dataType: 'float32', shape: ${scaleShape}}, new Float32Array([${scaleValues.join(', ')}]));`);

    const zpArr = zpValues ?? [0];
    const allZerosZp = zpArr.every((v) => v === 0);

    if (allZerosZp) {
      emitter.line(`const ${output} = builder.cast(builder.div(${input}, ${scaleVar}), '${outputDtype}');`);
    } else {
      const zpConstVar = `${output}_zpf`;
      const zpShape = zpArr.length === 1 ? '[]' : `[${zpArr.length}]`;
      emitter.line(`const ${zpConstVar} = builder.constant({dataType: 'float32', shape: ${zpShape}}, new Float32Array([${zpArr.map((v) => v.toString()).join(', ')}]));`);
      emitter.line(`const ${output} = builder.cast(builder.add(builder.div(${input}, ${scaleVar}), ${zpConstVar}), '${outputDtype}');`);
    }
  } else {
    emitter.comment('No quantization params found — using cast as fallback');
    emitter.line(`const ${output} = builder.cast(${input}, 'uint8');`);
  }
}

/** Find quantization scale from node attributes (set by parser as input_<idx>_scale) */
function findQuantizationScale(node: NodeIR): number[] | undefined {
  for (const key of Object.keys(node.attributes)) {
    if (key.endsWith('_scale')) {
      const val = node.attributes[key];
      if (Array.isArray(val)) return val as number[];
    }
  }
  return undefined;
}

/** Find quantization zero point from node attributes */
function findQuantizationZeroPoint(node: NodeIR): number[] | undefined {
  for (const key of Object.keys(node.attributes)) {
    if (key.endsWith('_zero_point')) {
      const val = node.attributes[key];
      if (Array.isArray(val)) return val as number[];
    }
  }
  return undefined;
}

function emitBroadcastTo(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // WebNN expand(input, newShape) — newShape is sequence<unsigned long>, not an MLOperand
  if (node.inputs.length > 1) {
    const shapeValues = emitter.constantIntValues(node.inputs[1]);
    if (shapeValues) {
      emitter.line(`const ${output} = builder.expand(${input}, [${shapeValues}]);`);
    } else {
      const outShape = emitter.tensorShape(node.outputs[0]);
      if (outShape) {
        emitter.line(`const ${output} = builder.expand(${input}, [${outShape.join(', ')}]);`);
      } else {
        emitter.comment('WARNING: BROADCAST_TO shape unknown');
        emitter.line(`const ${output} = ${input};`);
      }
    }
  } else {
    emitter.line(`const ${output} = ${input}; // BROADCAST_TO`);
  }
}

function emitWhere(node: NodeIR, emitter: CodeEmitter): void {
  const cond = emitter.ref(node.inputs[0]);
  const a = emitter.ref(node.inputs[1]);
  const b = emitter.ref(node.inputs[2]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.where(${cond}, ${a}, ${b});`);
}

function emitSelectV2(node: NodeIR, emitter: CodeEmitter): void {
  emitWhere(node, emitter);
}

function emitIdentity(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.identity(${input});`);
}

function emitAddN(node: NodeIR, emitter: CodeEmitter): void {
  const inputs = node.inputs.map((n) => emitter.ref(n));
  const output = emitter.declare(node.outputs[0]);

  if (inputs.length === 1) {
    emitter.line(`const ${output} = ${inputs[0]};`);
  } else {
    let result = `builder.add(${inputs[0]}, ${inputs[1]})`;
    for (let i = 2; i < inputs.length; i++) {
      result = `builder.add(${result}, ${inputs[i]})`;
    }
    emitter.line(`const ${output} = ${result};`);
  }
}

registerTfliteOp('CAST', emitCast);
registerTfliteOp('ARG_MAX', emitArgMax);
registerTfliteOp('ARG_MIN', emitArgMin);
registerTfliteOp('DEPTH_TO_SPACE', emitDepthToSpace);
registerTfliteOp('SPACE_TO_DEPTH', emitSpaceToDepth);
registerTfliteOp('CUMSUM', emitCumSum);
registerTfliteOp('DEQUANTIZE', emitDequantize);
registerTfliteOp('QUANTIZE', emitQuantize);
registerTfliteOp('BROADCAST_TO', emitBroadcastTo);
registerTfliteOp('WHERE', emitWhere);
registerTfliteOp('SELECT_V2', emitSelectV2);
registerTfliteOp('SELECT', emitSelectV2);
registerTfliteOp('ZEROS_LIKE', emitIdentity);
registerTfliteOp('ADD_N', emitAddN);
