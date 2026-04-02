// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/
// Miscellaneous ops: Resize, GatherElements, GatherND, ScatterElements, ScatterND,
// ArgMax, ArgMin, CumSum, DepthToSpace, SpaceToDepth, Trilu, Min, Max, Mean,
// Equal, Greater, GreaterOrEqual, Less, LessOrEqual, Not, LogicalAnd, LogicalOr, Xor

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp, registerOnnxOps } from '../registry.js';

// resize_op_builder.cc
function emitResize(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const mode = (node.attributes.mode as string) ?? 'nearest';
  const opts: string[] = [];

  if (mode === 'linear') {
    opts.push(`mode: 'linear'`);
  } else {
    opts.push(`mode: 'nearest-neighbor'`);
  }

  // Helper: check if input is an empty placeholder tensor (shape [0])
  const isEmpty = (idx: number) =>
    node.inputs.length <= idx || node.inputs[idx] === '' ||
    (emitter.isConstant(node.inputs[idx]) && emitter.constantShape(node.inputs[idx]).some(d => d === 0));

  const hasScales = !isEmpty(2);
  const hasSizes = !isEmpty(3);

  // Resolve axes — opset 18+ may specify axes attribute
  const axesAttr = node.attributes.axes as number[] | undefined;

  if (hasSizes) {
    // ORT reads sizes from constant initializer and extracts spatial dims only.
    // WebNN resample2d expects sizes as a JS array of [height, width].
    if (emitter.isConstant(node.inputs[3])) {
      const sizesValues = emitter.constantIntValues(node.inputs[3]);
      if (sizesValues) {
        if (axesAttr && axesAttr.length === 2) {
          // Opset 18+: sizes already contains 2 spatial values
          opts.push(`sizes: [${sizesValues.join(', ')}]`);
          opts.push(`axes: [${axesAttr.join(', ')}]`);
        } else if (sizesValues.length === 4) {
          // Opset 11-17: sizes has 4 elements [N, C, H, W], extract spatial dims
          opts.push(`sizes: [${sizesValues[2]}, ${sizesValues[3]}]`);
          opts.push(`axes: [2, 3]`);
        } else {
          opts.push(`sizes: [${sizesValues.join(', ')}]`);
        }
      } else {
        opts.push(`sizes: ${emitter.ref(node.inputs[3])}`);
      }
    } else {
      // Sizes is computed dynamically (e.g., from Shape → Slice → Concat chain).
      // Try to trace through a Concat node to find the constant spatial dims.
      let resolved = false;
      const producer = emitter.findProducerNode(node.inputs[3]);
      if (producer && producer.opType === 'Concat') {
        // Common pattern: Concat([shape[:2], [target_h, target_w]])
        // The last input to Concat is typically the constant spatial dims
        for (let i = producer.inputs.length - 1; i >= 0; i--) {
          if (emitter.isConstant(producer.inputs[i])) {
            const spatialValues = emitter.constantIntValues(producer.inputs[i]);
            if (spatialValues && spatialValues.length === 2) {
              opts.push(`sizes: [${spatialValues.join(', ')}]`);
              opts.push(`axes: [2, 3]`);
              resolved = true;
              break;
            }
          }
        }
      }
      if (!resolved) {
        // Fallback: try output shape
        const outputShape = emitter.tensorShape(node.outputs[0]);
        if (outputShape && outputShape.length === 4 &&
            typeof outputShape[2] === 'number' && typeof outputShape[3] === 'number') {
          opts.push(`sizes: [${outputShape[2]}, ${outputShape[3]}]`);
          opts.push(`axes: [2, 3]`);
        } else {
          emitter.comment(`WARNING: Resize sizes is dynamic and could not be resolved`);
          opts.push(`sizes: ${emitter.ref(node.inputs[3])}`);
        }
      }
    }
  } else if (hasScales) {
    if (emitter.isConstant(node.inputs[2])) {
      const scalesRaw = emitter.constantRawData(node.inputs[2]);
      if (scalesRaw && scalesRaw.byteLength > 0) {
        const aligned = new ArrayBuffer(scalesRaw.byteLength);
        new Uint8Array(aligned).set(scalesRaw);
        const scalesValues = Array.from(new Float32Array(aligned));
        if (axesAttr && axesAttr.length === 2) {
          opts.push(`scales: [${scalesValues.join(', ')}]`);
          opts.push(`axes: [${axesAttr.join(', ')}]`);
        } else if (scalesValues.length === 4) {
          opts.push(`scales: [${scalesValues[2]}, ${scalesValues[3]}]`);
          opts.push(`axes: [2, 3]`);
        } else {
          opts.push(`scales: [${scalesValues.join(', ')}]`);
        }
      } else {
        opts.push(`scales: ${emitter.ref(node.inputs[2])}`);
      }
    } else {
      opts.push(`scales: ${emitter.ref(node.inputs[2])}`);
    }
  }

  const coordinateTransformMode = node.attributes.coordinate_transformation_mode as string | undefined;
  if (coordinateTransformMode === 'align_corners') {
    opts.push(`alignCorners: true`);
  } else if (coordinateTransformMode === 'half_pixel') {
    opts.push(`halfPixelCenters: true`);
  }

  emitter.line(`const ${output} = builder.resample2d(${input}, { ${opts.join(', ')} });`);
}

// gather_elements_op_builder.cc
function emitGatherElements(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  const axis = (node.attributes.axis as number) ?? 0;
  emitter.line(`const ${output} = builder.gatherElements(${input}, ${indices}, { axis: ${axis} });`);
}

// gathernd_op_builder.cc
function emitGatherND(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.gatherND(${input}, ${indices});`);
}

// scatter_elements_op_builder.cc
function emitScatterElements(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const updates = emitter.ref(node.inputs[2]);
  const output = emitter.declare(node.outputs[0]);
  const axis = (node.attributes.axis as number) ?? 0;
  emitter.line(`const ${output} = builder.scatterElements(${input}, ${indices}, ${updates}, { axis: ${axis} });`);
}

// scatternd_op_builder.cc
function emitScatterND(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const updates = emitter.ref(node.inputs[2]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.scatterND(${input}, ${indices}, ${updates});`);
}

// argmax_min_op_builder.cc
function emitArgMaxMin(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const webnnOp = node.opType === 'ArgMax' ? 'argMax' : 'argMin';
  let axis = (node.attributes.axis as number) ?? 0;
  const keepdims = (node.attributes.keepdims as number) ?? 1;

  // WebNN axis is unsigned long — normalize negative values
  if (axis < 0) {
    const inputShape = emitter.tensorShape(node.inputs[0]);
    if (inputShape) axis += inputShape.length;
  }

  const opts: string[] = [];
  if (!keepdims) opts.push(`keepDimensions: false`);
  // ONNX ArgMax/ArgMin output int64 by default; WebNN defaults to int32.
  // Match the model's declared output type.
  const outputDtype = emitter.tensorDataType(node.outputs[0]);
  if (outputDtype && outputDtype !== 'int32') {
    opts.push(`outputDataType: '${outputDtype}'`);
  }

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.${webnnOp}(${input}, ${axis}${optsStr});`);
}

// cumsum_op_builder.cc
function emitCumSum(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const axis = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  const exclusive = (node.attributes.exclusive as number) ?? 0;
  const reverse = (node.attributes.reverse as number) ?? 0;

  const opts: string[] = [];
  if (exclusive) opts.push(`exclusive: true`);
  if (reverse) opts.push(`reversed: true`);

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.cumulativeSum(${input}, ${axis}${optsStr});`);
}

// depthToSpace_op_builder.cc — DepthToSpace
// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/depthToSpace_op_builder.cc
// WebNN does not have depthToSpace; ORT decomposes it into reshape → transpose → reshape.
function emitDepthToSpace(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const blocksize = (node.attributes.blocksize as number) ?? 1;
  const mode = (node.attributes.mode as string) ?? 'DCR';

  // Requires known input shape [batch, channels, height, width]
  const inputShape = emitter.tensorShape(node.inputs[0]);
  if (!inputShape || inputShape.length !== 4) {
    emitter.comment(`DepthToSpace: unknown input shape, cannot decompose`);
    return;
  }
  const [batch, channels, height, width] = inputShape;
  const newC = `${Number(channels) / (blocksize * blocksize)}`;
  const newH = `${Number(height) * blocksize}`;
  const newW = `${Number(width) * blocksize}`;

  let shape1: string;
  let perm: string;
  if (mode === 'CRD') {
    shape1 = `[${batch}, ${newC}, ${blocksize}, ${blocksize}, ${height}, ${width}]`;
    perm = `[0, 1, 4, 2, 5, 3]`;
  } else {
    // DCR (default)
    shape1 = `[${batch}, ${blocksize}, ${blocksize}, ${newC}, ${height}, ${width}]`;
    perm = `[0, 3, 4, 1, 5, 2]`;
  }
  const shape2 = `[${batch}, ${newC}, ${newH}, ${newW}]`;

  // Step 1: Reshape to 6D
  emitter.line(`const ${output}_r1 = builder.reshape(${input}, ${shape1});`);
  // Step 2: Transpose
  emitter.line(`const ${output}_t = builder.transpose(${output}_r1, { permutation: ${perm} });`);
  // Step 3: Reshape to output shape
  emitter.line(`const ${output} = builder.reshape(${output}_t, ${shape2});`);
}

// SpaceToDepth — inverse of DepthToSpace, decomposed into reshape → transpose → reshape.
function emitSpaceToDepth(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const blocksize = (node.attributes.blocksize as number) ?? 1;

  const inputShape = emitter.tensorShape(node.inputs[0]);
  if (!inputShape || inputShape.length !== 4) {
    emitter.comment(`SpaceToDepth: unknown input shape, cannot decompose`);
    return;
  }
  const [batch, channels, height, width] = inputShape;
  const newC = `${Number(channels) * blocksize * blocksize}`;
  const newH = `${Number(height) / blocksize}`;
  const newW = `${Number(width) / blocksize}`;

  // Reshape [b, c, h, w] → [b, c, h/bs, bs, w/bs, bs]
  const shape1 = `[${batch}, ${channels}, ${newH}, ${blocksize}, ${newW}, ${blocksize}]`;
  // Transpose → [b, c, bs, bs, h/bs, w/bs]
  const perm = `[0, 1, 3, 5, 2, 4]`;
  // Reshape → [b, c*bs*bs, h/bs, w/bs]
  const shape2 = `[${batch}, ${newC}, ${newH}, ${newW}]`;

  emitter.line(`const ${output}_r1 = builder.reshape(${input}, ${shape1});`);
  emitter.line(`const ${output}_t = builder.transpose(${output}_r1, { permutation: ${perm} });`);
  emitter.line(`const ${output} = builder.reshape(${output}_t, ${shape2});`);
}

// triangular_op_builder.cc
function emitTrilu(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const upper = (node.attributes.upper as number) ?? 1;
  const webnnOp = upper ? 'triangular' : 'triangular';
  const opts: string[] = [];
  if (!upper) opts.push(`upper: false`);
  if (node.inputs.length > 1 && node.inputs[1] !== '') {
    opts.push(`diagonal: ${emitter.ref(node.inputs[1])}`);
  }
  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.${webnnOp}(${input}${optsStr});`);
}

// logical_op_builder.cc
function emitLogicalBinary(node: NodeIR, emitter: CodeEmitter): void {
  const a = emitter.ref(node.inputs[0]);
  const b = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);

  const opMap: Record<string, string> = {
    Equal: 'equal',
    Greater: 'greater',
    GreaterOrEqual: 'greaterOrEqual',
    Less: 'lesser',
    LessOrEqual: 'lesserOrEqual',
  };
  const webnnOp = opMap[node.opType] ?? 'equal';
  emitter.line(`const ${output} = builder.${webnnOp}(${a}, ${b});`);
}

function emitLogicalNot(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.logicalNot(${input});`);
}

// min/max/mean → element-wise reduction via chain
function emitMinMaxMean(node: NodeIR, emitter: CodeEmitter): void {
  const output = emitter.declare(node.outputs[0]);
  const webnnOp = node.opType === 'Min' ? 'min' : node.opType === 'Max' ? 'max' : 'add';

  if (node.inputs.length === 2) {
    const a = emitter.ref(node.inputs[0]);
    const b = emitter.ref(node.inputs[1]);
    if (node.opType === 'Mean') {
      const sum = emitter.declare(`${node.outputs[0]}_sum`);
      emitter.line(`const ${sum} = builder.add(${a}, ${b});`);
      emitter.line(`const ${output} = builder.div(${sum}, builder.constant({ dataType: 'float32', shape: [] }, new Float32Array([${node.inputs.length}])));`);
    } else {
      emitter.line(`const ${output} = builder.${webnnOp}(${a}, ${b});`);
    }
  } else {
    // Chain multiple inputs
    let result = emitter.ref(node.inputs[0]);
    for (let i = 1; i < node.inputs.length; i++) {
      const next = emitter.ref(node.inputs[i]);
      const tmp = emitter.declare(`${node.outputs[0]}_chain_${i}`);
      emitter.line(`const ${tmp} = builder.${webnnOp}(${result}, ${next});`);
      result = tmp;
    }
    if (node.opType === 'Mean') {
      emitter.line(`const ${output} = builder.div(${result}, builder.constant({ dataType: 'float32', shape: [] }, new Float32Array([${node.inputs.length}])));`);
    } else {
      emitter.line(`const ${output} = ${result};`);
    }
  }
}

// qdq_op_builder.cc — DequantizeLinear, QuantizeLinear
// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/qdq_op_builder.cc
// WebNN signature: builder.dequantizeLinear(input, scale, zeroPoint, options?)
//                  builder.quantizeLinear(input, scale, zeroPoint, options?)

// WebNN dequantizeLinear only supports [int8, uint8, int4, uint4] as input types.
const WEBNN_DQL_SUPPORTED_TYPES = new Set(['int8', 'uint8', 'int4', 'uint4']);

function emitQDQ(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const scale = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  const hasZeroPoint = node.inputs.length > 2 && node.inputs[2] !== '';
  const webnnOp = node.opType === 'DequantizeLinear' ? 'dequantizeLinear' : 'quantizeLinear';

  // DEVIATION: WebNN dequantizeLinear does not support int32 (or float16/float32) inputs,
  // but ONNX DequantizeLinear does (e.g. output of MatMulInteger/ConvInteger).
  // Decompose manually: output = (cast(input, float32) - cast(zeroPoint, float32)) * scale
  const inputDtype = emitter.tensorDataType(node.inputs[0]);
  if (node.opType === 'DequantizeLinear' && inputDtype && !WEBNN_DQL_SUPPORTED_TYPES.has(inputDtype)) {
    emitter.comment(`DequantizeLinear decomposed: input type '${inputDtype}' not supported by WebNN dequantizeLinear`);
    const castInput = `${output}_cast_input`;
    emitter.line(`const ${castInput} = builder.cast(${input}, 'float32');`);
    if (hasZeroPoint) {
      const castZp = `${output}_cast_zp`;
      emitter.line(`const ${castZp} = builder.cast(${emitter.ref(node.inputs[2])}, 'float32');`);
      emitter.line(`const ${output} = builder.mul(builder.sub(${castInput}, ${castZp}), ${scale});`);
    } else {
      emitter.line(`const ${output} = builder.mul(${castInput}, ${scale});`);
    }
    return;
  }

  const blockSize = (node.attributes.block_size as number) ?? 0;
  let axis = (node.attributes.axis as number) ?? 1;

  // Get input shape to determine rank for reshaping.
  // QDQ ops preserve shape, so fall back to output shape if input shape is unknown.
  const inputShape = emitter.tensorShape(node.inputs[0]) ?? emitter.tensorShape(node.outputs[0]);
  const inputRank = inputShape ? inputShape.length : 0;
  if (axis < 0 && inputRank > 0) {
    axis = axis + inputRank;
  }

  // ORT qdq_op_builder.cc: WebNN requires scale and zeroPoint to have the same rank as input.
  // For per-tensor quantization (scalar scale), reshape to [1,1,...,1].
  // For per-axis quantization (1D scale), reshape to [1,...,N,...,1] with N at the axis position.
  let scaleRef = scale;
  let zpRef: string | undefined;
  let zpShape: string | undefined; // track shape for synthesized zeroPoint

  const scaleShape = emitter.isConstant(node.inputs[1]) ? emitter.constantShape(node.inputs[1]) : null;
  const scaleShapeFromGraph = emitter.tensorShape(node.inputs[1]);
  const effectiveScaleRank = scaleShape ? scaleShape.length : (scaleShapeFromGraph ? scaleShapeFromGraph.length : -1);

  if (effectiveScaleRank === 0 && inputRank > 0) {
    // Scalar scale — reshape to all-ones shape matching input rank.
    const targetShape = Array(inputRank).fill(1);
    scaleRef = `${output}_scale_reshaped`;
    emitter.line(`const ${scaleRef} = builder.reshape(${scale}, [${targetShape.join(', ')}]);`);
    zpShape = `[${targetShape.join(', ')}]`;

    if (hasZeroPoint) {
      zpRef = `${output}_zp_reshaped`;
      emitter.line(`const ${zpRef} = builder.reshape(${emitter.ref(node.inputs[2])}, [${targetShape.join(', ')}]);`);
    }
  } else if (effectiveScaleRank === 1 && inputRank > 1 &&
             blockSize === 0) {
    // Per-axis quantization: reshape scale to [1,...,scaleSize,...,1] to match input rank.
    // WebNN requires scale rank == input rank (Chromium validates this strictly).
    const scaleSize = scaleShape ? scaleShape[0] : (scaleShapeFromGraph ? scaleShapeFromGraph[0] : null);
    if (scaleSize !== null) {
      const targetShape = Array(inputRank).fill(1);
      targetShape[axis] = scaleSize;
      scaleRef = `${output}_scale_reshaped`;
      emitter.line(`const ${scaleRef} = builder.reshape(${scale}, [${targetShape.join(', ')}]);`);
      zpShape = `[${targetShape.join(', ')}]`;

      if (hasZeroPoint) {
        zpRef = `${output}_zp_reshaped`;
        emitter.line(`const ${zpRef} = builder.reshape(${emitter.ref(node.inputs[2])}, [${targetShape.join(', ')}]);`);
      }
    }
  }

  if (hasZeroPoint && !zpRef) {
    zpRef = emitter.ref(node.inputs[2]);
  }

  if (!hasZeroPoint) {
    // ORT: Create a zero constant with matching type and same shape as (reshaped) scale.
    // DequantizeLinear: zeroPoint type matches input type
    // QuantizeLinear: zeroPoint type matches output type
    const outputDtype = emitter.tensorDataType(node.outputs[0]);
    const zpType = node.opType === 'DequantizeLinear' ? (inputDtype ?? 'uint8') : (outputDtype ?? 'uint8');
    const arrayType = dataTypeToTypedArray(zpType);
    const shape = zpShape ?? (scaleShape ? `[${scaleShape.join(', ')}]` : '[]');
    zpRef = `${output}_zp`;
    emitter.line(`const ${zpRef} = builder.constant({dataType: '${zpType}', shape: ${shape}}, new ${arrayType}([0]));`);
  }

  emitter.line(`const ${output} = builder.${webnnOp}(${input}, ${scaleRef}, ${zpRef});`);
}

function dataTypeToTypedArray(dt: string): string {
  switch (dt) {
    case 'float32': return 'Float32Array';
    case 'float16': return 'Float16Array';
    case 'int8': return 'Int8Array';
    case 'uint8': return 'Uint8Array';
    case 'int32': return 'Int32Array';
    case 'uint32': return 'Uint32Array';
    case 'int64': return 'BigInt64Array';
    case 'uint64': return 'BigUint64Array';
    default: return 'Uint8Array';
  }
}

registerOnnxOp('Resize', emitResize);
registerOnnxOp('GatherElements', emitGatherElements);
registerOnnxOp('GatherND', emitGatherND);
registerOnnxOp('ScatterElements', emitScatterElements);
registerOnnxOp('ScatterND', emitScatterND);
registerOnnxOp('ArgMax', emitArgMaxMin);
registerOnnxOp('ArgMin', emitArgMaxMin);
registerOnnxOp('CumSum', emitCumSum);
registerOnnxOp('DepthToSpace', emitDepthToSpace);
registerOnnxOp('SpaceToDepth', emitSpaceToDepth);
registerOnnxOp('Trilu', emitTrilu);
registerOnnxOps(['Equal', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual'], emitLogicalBinary);
registerOnnxOp('Not', emitLogicalNot);
registerOnnxOps(['Min', 'Max', 'Mean'], emitMinMaxMean);
registerOnnxOps(['DequantizeLinear', 'QuantizeLinear'], emitQDQ);

// logical_op_builder.cc — And, Or, Xor, IsInf, IsNaN
function emitLogicalBinaryBool(node: NodeIR, emitter: CodeEmitter): void {
  const a = emitter.ref(node.inputs[0]);
  const b = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  const opMap: Record<string, string> = { And: 'logicalAnd', Or: 'logicalOr', Xor: 'logicalXor' };
  emitter.line(`const ${output} = builder.${opMap[node.opType]}(${a}, ${b});`);
}
registerOnnxOps(['And', 'Or', 'Xor'], emitLogicalBinaryBool);

function emitIsInf(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const detectPositive = (node.attributes.detect_positive as number) ?? 1;
  const detectNegative = (node.attributes.detect_negative as number) ?? 1;
  if (detectPositive && detectNegative) {
    emitter.line(`const ${output} = builder.isInfinite(${input});`);
  } else if (detectPositive) {
    emitter.line(`const ${output}_inf = builder.isInfinite(${input});`);
    emitter.line(`const ${output}_pos = builder.greater(${input}, builder.constant({dataType: 'float32', shape: []}, new Float32Array([0])));`);
    emitter.line(`const ${output} = builder.logicalAnd(${output}_inf, ${output}_pos);`);
  } else if (detectNegative) {
    emitter.line(`const ${output}_inf = builder.isInfinite(${input});`);
    emitter.line(`const ${output}_neg = builder.lesser(${input}, builder.constant({dataType: 'float32', shape: []}, new Float32Array([0])));`);
    emitter.line(`const ${output} = builder.logicalAnd(${output}_inf, ${output}_neg);`);
  } else {
    emitter.line(`const ${output}_inf = builder.isInfinite(${input});`);
    emitter.line(`const ${output} = builder.logicalAnd(${output}_inf, builder.logicalNot(${output}_inf));`);
  }
}
registerOnnxOp('IsInf', emitIsInf);

function emitIsNaN(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.isNaN(${input});`);
}
registerOnnxOp('IsNaN', emitIsNaN);

// dynamicQuantizeLinear_op_builder.cc
// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/dynamicQuantizeLinear_op_builder.cc
// Decomposes to: reduceMin/Max → scale computation → roundEven(zeroPoint) → cast → quantizeLinear
function emitDynamicQuantizeLinear(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const y = emitter.declare(node.outputs[0]);
  const yScale = node.outputs.length > 1 && node.outputs[1] !== '' ? emitter.declare(node.outputs[1]) : null;
  const yZp = node.outputs.length > 2 && node.outputs[2] !== '' ? emitter.declare(node.outputs[2]) : null;
  emitter.comment('DynamicQuantizeLinear');

  // Q_Min = 0, Q_Max = 255
  const qMin = `${y}_q_min`;
  const qMax = `${y}_q_max`;
  emitter.line(`const ${qMin} = builder.constant({dataType: 'float32', shape: []}, new Float32Array([0]));`);
  emitter.line(`const ${qMax} = builder.constant({dataType: 'float32', shape: []}, new Float32Array([255]));`);

  // X_Min = ReduceMin(x), X_Min_Adjusted = Min(X_Min, Q_Min)
  const xMin = `${y}_x_min`;
  emitter.line(`const ${xMin} = builder.reduceMin(${input});`);
  const xMinAdj = `${y}_x_min_adj`;
  emitter.line(`const ${xMinAdj} = builder.min(${xMin}, ${qMin});`);

  // X_Max = ReduceMax(x), X_Max_Adjusted = Max(X_Max, Q_Min)
  const xMax = `${y}_x_max`;
  emitter.line(`const ${xMax} = builder.reduceMax(${input});`);
  const xMaxAdj = `${y}_x_max_adj`;
  emitter.line(`const ${xMaxAdj} = builder.max(${xMax}, ${qMin});`);

  // Scale = (X_Max_Adjusted - X_Min_Adjusted) / 255
  const xRange = `${y}_range`;
  emitter.line(`const ${xRange} = builder.sub(${xMaxAdj}, ${xMinAdj});`);
  const scale = `${y}_scale`;
  emitter.line(`const ${scale} = builder.div(${xRange}, ${qMax});`);

  // Initial_ZeroPoint = Q_Min - X_Min_Adjusted / Scale
  // Clipped = Clamp(Initial_ZeroPoint, 0, 255)
  // Rounded = RoundEven(Clipped)
  // ZeroPoint = Cast(Rounded, 'uint8')
  const minScaled = `${y}_min_scaled`;
  emitter.line(`const ${minScaled} = builder.div(${xMinAdj}, ${scale});`);
  const initZp = `${y}_init_zp`;
  emitter.line(`const ${initZp} = builder.sub(${qMin}, ${minScaled});`);
  const clippedZp = `${y}_clipped_zp`;
  emitter.line(`const ${clippedZp} = builder.clamp(${initZp}, { minValue: 0, maxValue: 255 });`);
  const roundedZp = `${y}_rounded_zp`;
  emitter.line(`const ${roundedZp} = builder.roundEven(${clippedZp});`);
  const zp = `${y}_zp`;
  emitter.line(`const ${zp} = builder.cast(${roundedZp}, 'uint8');`);

  // ORT: WebNN quantizeLinear requires scale and zeroPoint to have the same rank as input.
  // The scale and zeroPoint are scalar (from reduceMin/reduceMax), so reshape to [1,1,...,1].
  const inputShape = emitter.tensorShape(node.inputs[0]);
  const inputRank = inputShape ? inputShape.length : 0;
  let scaleForQL = scale;
  let zpForQL = zp;
  if (inputRank > 0) {
    const targetShape = Array(inputRank).fill(1);
    scaleForQL = `${y}_scale_reshaped`;
    emitter.line(`const ${scaleForQL} = builder.reshape(${scale}, [${targetShape.join(', ')}]);`);
    zpForQL = `${y}_zp_reshaped`;
    emitter.line(`const ${zpForQL} = builder.reshape(${zp}, [${targetShape.join(', ')}]);`);
  }

  // y = quantizeLinear(x, scale, zeroPoint)
  emitter.line(`const ${y} = builder.quantizeLinear(${input}, ${scaleForQL}, ${zpForQL});`);
  // Output the original (non-reshaped) scale and zeroPoint as the DynamicQuantizeLinear outputs
  if (yScale) emitter.line(`const ${yScale} = ${scale};`);
  if (yZp) emitter.line(`const ${yZp} = ${zp};`);
}
registerOnnxOp('DynamicQuantizeLinear', emitDynamicQuantizeLinear);

// lrn_op_builder.cc — Local Response Normalization
function emitLRN(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const alpha = (node.attributes.alpha as number) ?? 0.0001;
  const beta = (node.attributes.beta as number) ?? 0.75;
  const bias = (node.attributes.bias as number) ?? 1.0;
  const size = (node.attributes.size as number) ?? 1;
  emitter.comment(`LRN — size=${size}, alpha=${alpha}, beta=${beta}, bias=${bias}`);
  // Decompose: y = x / (bias + alpha/size * sum(x^2, local_window))^beta
  const sq = `${output}_sq`;
  const alphaConst = `${output}_alpha`;
  const biasConst = `${output}_bias`;
  const betaConst = `${output}_beta`;
  emitter.line(`const ${sq} = builder.mul(${input}, ${input});`);
  emitter.line(`const ${alphaConst} = builder.constant({dataType: 'float32', shape: []}, new Float32Array([${alpha / size}]));`);
  emitter.line(`const ${biasConst} = builder.constant({dataType: 'float32', shape: []}, new Float32Array([${bias}]));`);
  emitter.line(`const ${betaConst} = builder.constant({dataType: 'float32', shape: []}, new Float32Array([${beta}]));`);
  // Use averagePool as a local sum approximation
  const pooled = `${output}_pooled`;
  emitter.line(`const ${pooled} = builder.averagePool2d(${sq}, { windowDimensions: [1, ${size}], autoPad: 'same-upper' });`);
  const norm = `${output}_norm`;
  emitter.line(`const ${norm} = builder.pow(builder.add(${biasConst}, builder.mul(${alphaConst}, ${pooled})), ${betaConst});`);
  emitter.line(`const ${output} = builder.div(${input}, ${norm});`);
}
registerOnnxOp('LRN', emitLRN);

// pool_op_builder.cc — GlobalLpPool
function emitGlobalLpPool(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const p = (node.attributes.p as number) ?? 2;
  emitter.comment(`GlobalLpPool (p=${p})`);
  if (p === 2) {
    emitter.line(`const ${output} = builder.l2Pool2d(${input});`);
  } else {
    // General case: (sum(|x|^p))^(1/p)
    const abs = `${output}_abs`;
    const pow = `${output}_pow`;
    emitter.line(`const ${abs} = builder.abs(${input});`);
    const pConst = `${output}_p`;
    emitter.line(`const ${pConst} = builder.constant({dataType: 'float32', shape: []}, new Float32Array([${p}]));`);
    emitter.line(`const ${pow} = builder.pow(${abs}, ${pConst});`);
    const sum = `${output}_sum`;
    emitter.line(`const ${sum} = builder.reduceSum(${pow}, { axes: [2, 3], keepDimensions: true });`);
    const invP = `${output}_invp`;
    emitter.line(`const ${invP} = builder.constant({dataType: 'float32', shape: []}, new Float32Array([${1 / p}]));`);
    emitter.line(`const ${output} = builder.pow(${sum}, ${invP});`);
  }
}
registerOnnxOp('GlobalLpPool', emitGlobalLpPool);
