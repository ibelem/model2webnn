// TFLite miscellaneous ops: CAST, ARG_MAX, ARG_MIN, DEPTH_TO_SPACE, SPACE_TO_DEPTH,
// CUMSUM, DEQUANTIZE, QUANTIZE, BROADCAST_TO, WHERE, SELECT_V2, SELECT,
// ZEROS_LIKE, ADD_N, IDENTITY

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';
import { dataTypeToArray } from './common.js';

function emitCast(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const outType = (node.attributes.out_data_type as string) ?? 'FLOAT32';
  emitter.line(`const ${output} = builder.cast(${input}, '${outType.toLowerCase()}');`);
}

function emitArgMax(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.argMax(${input}, { keepDimensions: true });`);
}

function emitArgMin(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.argMin(${input}, { keepDimensions: true });`);
}

function emitDepthToSpace(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.depthToSpace(${input});`);
}

function emitSpaceToDepth(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.spaceToDepth(${input});`);
}

function emitCumSum(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.cumulativeSum(${input}, 0);`);
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
    // Determine input rank and dtype for proper shape and type matching
    const inputShape = emitter.tensorShape(node.inputs[0]);
    const inputRank = inputShape ? inputShape.length : 0;
    const inputDtype = emitter.tensorDataType(node.inputs[0]) ?? 'int8';

    // Build shape that matches input rank (spec requirement)
    let paramShape: string;
    if (scaleValues.length === 1 && inputRank > 0) {
      paramShape = `[${new Array(inputRank).fill(1).join(', ')}]`;
    } else if (scaleValues.length > 1 && inputRank > 0) {
      const dims = new Array(inputRank).fill(1);
      dims[inputRank - 1] = scaleValues.length;
      paramShape = `[${dims.join(', ')}]`;
    } else {
      paramShape = `[${scaleValues.length}]`;
    }

    const scaleVar = `${output}_scale`;
    emitter.line(`const ${scaleVar} = builder.constant({dataType: 'float32', shape: ${paramShape}}, new Float32Array([${scaleValues.join(', ')}]));`);

    // zeroPoint must match input's dataType (spec requirement)
    const zpArr = zpValues ?? [0];
    const zpArrayType = dataTypeToArray[inputDtype] ?? 'Int8Array';
    const zpVar = `${output}_zp`;
    emitter.line(`const ${zpVar} = builder.constant({dataType: '${inputDtype}', shape: ${paramShape}}, new ${zpArrayType}([${zpArr.join(', ')}]));`);

    emitter.line(`const ${output} = builder.dequantizeLinear(${input}, ${scaleVar}, ${zpVar});`);
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
    // Determine input rank and output dtype for proper shape and type matching
    const inputShape = emitter.tensorShape(node.inputs[0]);
    const inputRank = inputShape ? inputShape.length : 0;
    const outputDtype = emitter.tensorDataType(node.outputs[0]) ?? 'uint8';

    // Build shape that matches input rank (spec requirement)
    let paramShape: string;
    if (scaleValues.length === 1 && inputRank > 0) {
      paramShape = `[${new Array(inputRank).fill(1).join(', ')}]`;
    } else if (scaleValues.length > 1 && inputRank > 0) {
      const dims = new Array(inputRank).fill(1);
      dims[inputRank - 1] = scaleValues.length;
      paramShape = `[${dims.join(', ')}]`;
    } else {
      paramShape = `[${scaleValues.length}]`;
    }

    const scaleVar = `${output}_scale`;
    emitter.line(`const ${scaleVar} = builder.constant({dataType: 'float32', shape: ${paramShape}}, new Float32Array([${scaleValues.join(', ')}]));`);

    // zeroPoint must match output's dataType (spec requirement)
    const zpArr = zpValues ?? [0];
    const zpArrayType = dataTypeToArray[outputDtype] ?? 'Uint8Array';
    const zpVar = `${output}_zp`;
    emitter.line(`const ${zpVar} = builder.constant({dataType: '${outputDtype}', shape: ${paramShape}}, new ${zpArrayType}([${zpArr.join(', ')}]));`);

    emitter.line(`const ${output} = builder.quantizeLinear(${input}, ${scaleVar}, ${zpVar});`);
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
  if (node.inputs.length > 1) {
    const shape = emitter.ref(node.inputs[1]);
    emitter.line(`const ${output} = builder.expand(${input}, ${shape});`);
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
