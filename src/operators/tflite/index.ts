// TFLite operator builders — maps TFLite ops to WebNN builder calls
// TFLite ops use NHWC layout by default (no layout conversion per project rules).

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';

// ──────────────────────────────────────────────────────
// Helper: emit fused activation after an op
// ──────────────────────────────────────────────────────
function emitFusedActivation(varName: string, activation: string | undefined, emitter: CodeEmitter): string {
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

// ──────────────────────────────────────────────────────
// Convolution ops
// ──────────────────────────────────────────────────────

function emitConv2D(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const filter = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  const bias = node.inputs.length > 2 ? emitter.ref(node.inputs[2]) : undefined;

  const attrs = node.attributes;
  const strides = [attrs.stride_h ?? 1, attrs.stride_w ?? 1];
  const dilations = [attrs.dilation_h_factor ?? 1, attrs.dilation_w_factor ?? 1];
  const padding = attrs.padding as string;

  emitter.comment(`Conv2D — padding: ${padding}`);

  const opts: string[] = [];
  opts.push(`strides: [${strides}]`);
  opts.push(`dilations: [${dilations}]`);
  if (padding === 'SAME') {
    opts.push(`autoPad: 'same-upper'`);
  }
  opts.push(`inputLayout: 'nhwc'`);
  opts.push(`filterLayout: 'ohwi'`);
  if (bias) {
    opts.push(`bias: ${bias}`);
  }

  const rawVar = `${output}_raw`;
  emitter.line(`const ${rawVar} = builder.conv2d(${input}, ${filter}, { ${opts.join(', ')} });`);

  const activation = attrs.fused_activation as string | undefined;
  const resultVar = emitFusedActivation(rawVar, activation, emitter);

  if (resultVar !== output) {
    emitter.line(`const ${output} = ${resultVar};`);
  }
}

function emitDepthwiseConv2D(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const filter = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  const bias = node.inputs.length > 2 ? emitter.ref(node.inputs[2]) : undefined;

  const attrs = node.attributes;
  const strides = [attrs.stride_h ?? 1, attrs.stride_w ?? 1];
  const dilations = [attrs.dilation_h_factor ?? 1, attrs.dilation_w_factor ?? 1];
  const padding = attrs.padding as string;

  emitter.comment(`DepthwiseConv2D — padding: ${padding}`);

  const opts: string[] = [];
  opts.push(`strides: [${strides}]`);
  opts.push(`dilations: [${dilations}]`);
  if (padding === 'SAME') {
    opts.push(`autoPad: 'same-upper'`);
  }
  opts.push(`inputLayout: 'nhwc'`);
  opts.push(`filterLayout: 'ihwo'`);
  // groups = input channels for depthwise
  opts.push(`groups: ${filter}.shape[3]`);
  if (bias) {
    opts.push(`bias: ${bias}`);
  }

  const rawVar = `${output}_raw`;
  emitter.line(`const ${rawVar} = builder.conv2d(${input}, ${filter}, { ${opts.join(', ')} });`);

  const activation = attrs.fused_activation as string | undefined;
  const resultVar = emitFusedActivation(rawVar, activation, emitter);

  if (resultVar !== output) {
    emitter.line(`const ${output} = ${resultVar};`);
  }
}

function emitTransposeConv(node: NodeIR, emitter: CodeEmitter): void {
  // TFLite TRANSPOSE_CONV inputs: output_shape, filter, input [, bias]
  const filter = emitter.ref(node.inputs[1]);
  const input = emitter.ref(node.inputs[2]);
  const output = emitter.declare(node.outputs[0]);
  const bias = node.inputs.length > 3 ? emitter.ref(node.inputs[3]) : undefined;

  const attrs = node.attributes;
  const strides = [attrs.stride_h ?? 1, attrs.stride_w ?? 1];
  const padding = attrs.padding as string;

  emitter.comment('TransposeConv');

  const opts: string[] = [];
  opts.push(`strides: [${strides}]`);
  if (padding === 'SAME') {
    opts.push(`autoPad: 'same-upper'`);
  }
  opts.push(`inputLayout: 'nhwc'`);
  opts.push(`filterLayout: 'ohwi'`);
  if (bias) {
    opts.push(`bias: ${bias}`);
  }

  emitter.line(`const ${output} = builder.convTranspose2d(${input}, ${filter}, { ${opts.join(', ')} });`);
}

// ──────────────────────────────────────────────────────
// Pooling ops
// ──────────────────────────────────────────────────────

function emitPool2D(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const attrs = node.attributes;
  const windowDims = [attrs.filter_height ?? 1, attrs.filter_width ?? 1];
  const strides = [attrs.stride_h ?? 1, attrs.stride_w ?? 1];
  const padding = attrs.padding as string;

  const webnnOp = node.opType === 'MAX_POOL_2D' ? 'maxPool2d' : 'averagePool2d';

  emitter.comment(`${node.opType}`);

  const opts: string[] = [];
  opts.push(`windowDimensions: [${windowDims}]`);
  opts.push(`strides: [${strides}]`);
  if (padding === 'SAME') {
    opts.push(`autoPad: 'same-upper'`);
  }
  opts.push(`layout: 'nhwc'`);

  const rawVar = `${output}_raw`;
  emitter.line(`const ${rawVar} = builder.${webnnOp}(${input}, { ${opts.join(', ')} });`);

  const activation = attrs.fused_activation as string | undefined;
  const resultVar = emitFusedActivation(rawVar, activation, emitter);

  if (resultVar !== output) {
    emitter.line(`const ${output} = ${resultVar};`);
  }
}

function emitL2Pool2D(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const attrs = node.attributes;
  const windowDims = [attrs.filter_height ?? 1, attrs.filter_width ?? 1];
  const strides = [attrs.stride_h ?? 1, attrs.stride_w ?? 1];
  const padding = attrs.padding as string;

  emitter.comment('L2_POOL_2D');

  const opts: string[] = [];
  opts.push(`windowDimensions: [${windowDims}]`);
  opts.push(`strides: [${strides}]`);
  if (padding === 'SAME') {
    opts.push(`autoPad: 'same-upper'`);
  }
  opts.push(`layout: 'nhwc'`);

  emitter.line(`const ${output} = builder.l2Pool2d(${input}, { ${opts.join(', ')} });`);
}

// ──────────────────────────────────────────────────────
// Activation ops
// ──────────────────────────────────────────────────────

function emitSimpleActivation(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  switch (node.opType) {
    case 'RELU':
      emitter.line(`const ${output} = builder.relu(${input});`);
      break;
    case 'RELU6':
      emitter.line(`const ${output} = builder.clamp(${input}, { minValue: 0, maxValue: 6 });`);
      break;
    case 'RELU_N1_TO_1':
      emitter.line(`const ${output} = builder.clamp(${input}, { minValue: -1, maxValue: 1 });`);
      break;
    case 'RELU_0_TO_1':
      emitter.line(`const ${output} = builder.clamp(${input}, { minValue: 0, maxValue: 1 });`);
      break;
    case 'LOGISTIC':
      emitter.line(`const ${output} = builder.sigmoid(${input});`);
      break;
    case 'TANH':
      emitter.line(`const ${output} = builder.tanh(${input});`);
      break;
    case 'ELU':
      emitter.line(`const ${output} = builder.elu(${input});`);
      break;
    case 'HARD_SWISH':
      emitter.line(`const ${output} = builder.hardSwish(${input});`);
      break;
    case 'LEAKY_RELU': {
      const alpha = (node.attributes.alpha as number) ?? 0.01;
      emitter.line(`const ${output} = builder.leakyRelu(${input}, { alpha: ${alpha} });`);
      break;
    }
    case 'GELU':
      emitter.line(`const ${output} = builder.gelu(${input});`);
      break;
    case 'PRELU': {
      const slope = emitter.ref(node.inputs[1]);
      emitter.line(`const ${output} = builder.prelu(${input}, ${slope});`);
      break;
    }
    default:
      emitter.comment(`Unsupported activation: ${node.opType}`);
      emitter.line(`const ${output} = ${input}; // UNSUPPORTED: ${node.opType}`);
  }
}

// ──────────────────────────────────────────────────────
// Element-wise binary ops
// ──────────────────────────────────────────────────────

function emitBinaryOp(node: NodeIR, emitter: CodeEmitter): void {
  const a = emitter.ref(node.inputs[0]);
  const b = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);

  const opMap: Record<string, string> = {
    ADD: 'add',
    SUB: 'sub',
    MUL: 'mul',
    DIV: 'div',
    POW: 'pow',
    MAXIMUM: 'max',
    MINIMUM: 'min',
    FLOOR_DIV: 'div',  // floor(a / b)
  };

  const webnnOp = opMap[node.opType];
  if (!webnnOp) {
    emitter.comment(`Unsupported binary op: ${node.opType}`);
    emitter.line(`const ${output} = ${a}; // UNSUPPORTED: ${node.opType}`);
    return;
  }

  emitter.comment(node.opType);

  if (node.opType === 'FLOOR_DIV') {
    emitter.line(`const ${output} = builder.floor(builder.div(${a}, ${b}));`);
  } else {
    const rawVar = `${output}_raw`;
    emitter.line(`const ${rawVar} = builder.${webnnOp}(${a}, ${b});`);

    const activation = node.attributes.fused_activation as string | undefined;
    const resultVar = emitFusedActivation(rawVar, activation, emitter);
    if (resultVar !== output) {
      emitter.line(`const ${output} = ${resultVar};`);
    }
  }
}

// ──────────────────────────────────────────────────────
// Unary math ops
// ──────────────────────────────────────────────────────

function emitUnaryOp(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const opMap: Record<string, string> = {
    ABS: 'abs',
    CEIL: 'ceil',
    COS: 'cos',
    EXP: 'exp',
    FLOOR: 'floor',
    LOG: 'log',
    NEG: 'neg',
    SIN: 'sin',
    SQRT: 'sqrt',
    RSQRT: 'reciprocal', // rsqrt(x) = 1/sqrt(x)
    ROUND: 'identity',   // placeholder — WebNN doesn't have round
    SIGN: 'sign',
    SQUARE: 'pow',        // x^2
  };

  if (node.opType === 'RSQRT') {
    emitter.comment('RSQRT → reciprocal(sqrt(x))');
    emitter.line(`const ${output} = builder.reciprocal(builder.sqrt(${input}));`);
  } else if (node.opType === 'SQUARE') {
    emitter.comment('SQUARE → pow(x, 2)');
    const twoConst = `${output}_two`;
    emitter.line(`const ${twoConst} = builder.constant({ dataType: 'float32', shape: [] }, new Float32Array([2]));`);
    emitter.line(`const ${output} = builder.pow(${input}, ${twoConst});`);
  } else {
    const webnnOp = opMap[node.opType];
    if (webnnOp && webnnOp !== 'identity') {
      emitter.line(`const ${output} = builder.${webnnOp}(${input});`);
    } else {
      emitter.line(`const ${output} = ${input}; // ${node.opType} (identity/passthrough)`);
    }
  }
}

// ──────────────────────────────────────────────────────
// Logical / comparison ops
// ──────────────────────────────────────────────────────

function emitLogicalOp(node: NodeIR, emitter: CodeEmitter): void {
  const output = emitter.declare(node.outputs[0]);

  const binaryOps: Record<string, string> = {
    EQUAL: 'equal',
    NOT_EQUAL: 'equal',  // negate after
    GREATER: 'greater',
    GREATER_EQUAL: 'greaterOrEqual',
    LESS: 'lesser',
    LESS_EQUAL: 'lesserOrEqual',
    LOGICAL_AND: 'logicalAnd',
    LOGICAL_OR: 'logicalOr',
  };

  if (node.opType === 'LOGICAL_NOT') {
    const input = emitter.ref(node.inputs[0]);
    emitter.line(`const ${output} = builder.logicalNot(${input});`);
    return;
  }

  const a = emitter.ref(node.inputs[0]);
  const b = emitter.ref(node.inputs[1]);

  if (node.opType === 'NOT_EQUAL') {
    emitter.line(`const ${output} = builder.logicalNot(builder.equal(${a}, ${b}));`);
  } else {
    const webnnOp = binaryOps[node.opType];
    if (webnnOp) {
      emitter.line(`const ${output} = builder.${webnnOp}(${a}, ${b});`);
    } else {
      emitter.comment(`Unsupported logical op: ${node.opType}`);
      emitter.line(`const ${output} = ${a}; // UNSUPPORTED: ${node.opType}`);
    }
  }
}

// ──────────────────────────────────────────────────────
// Shape manipulation ops
// ──────────────────────────────────────────────────────

function emitReshape(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  // TFLite: new_shape comes from options or second input tensor
  const newShape = node.attributes.new_shape as number[] | undefined;
  if (newShape) {
    emitter.line(`const ${output} = builder.reshape(${input}, [${newShape}]);`);
  } else if (node.inputs.length > 1) {
    const shapeInput = emitter.ref(node.inputs[1]);
    emitter.line(`const ${output} = builder.reshape(${input}, ${shapeInput});`);
  } else {
    emitter.comment('Reshape: no shape info available');
    emitter.line(`const ${output} = ${input};`);
  }
}

function emitSqueeze(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const squeezeDims = node.attributes.squeeze_dims as number[] | undefined;
  if (squeezeDims && squeezeDims.length > 0) {
    emitter.line(`const ${output} = builder.reshape(${input}, /* squeeze dims: [${squeezeDims}] */);`);
  } else {
    emitter.line(`const ${output} = builder.squeeze(${input});`);
  }
}

function emitExpandDims(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // axis comes from the second input tensor in TFLite
  emitter.comment('EXPAND_DIMS — unsqueeze along axis from second input');
  emitter.line(`const ${output} = builder.unsqueeze(${input});`);
}

function emitTranspose(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  if (node.inputs.length > 1) {
    const perm = emitter.ref(node.inputs[1]);
    emitter.line(`const ${output} = builder.transpose(${input}, { permutation: ${perm} });`);
  } else {
    emitter.line(`const ${output} = builder.transpose(${input});`);
  }
}

function emitConcatenation(node: NodeIR, emitter: CodeEmitter): void {
  const inputs = node.inputs.map((n) => emitter.ref(n));
  const output = emitter.declare(node.outputs[0]);
  const axis = (node.attributes.axis as number) ?? 0;

  emitter.line(`const ${output} = builder.concat([${inputs.join(', ')}], ${axis});`);
}

function emitSplit(node: NodeIR, emitter: CodeEmitter): void {
  // TFLite SPLIT: input(0) = split axis tensor, input(1) = data
  // The axis is from the first input (constant), num_splits from options
  const axisInput = emitter.ref(node.inputs[0]);
  const data = emitter.ref(node.inputs[1]);
  const numSplits = (node.attributes.num_splits as number) ?? node.outputs.length;

  emitter.comment(`SPLIT into ${numSplits} parts`);

  // Emit split outputs
  for (let i = 0; i < node.outputs.length; i++) {
    const out = emitter.declare(node.outputs[i]);
    emitter.line(`const ${out} = builder.split(${data}, ${numSplits}, { axis: ${axisInput} })[${i}];`);
  }
}

function emitSplitV(node: NodeIR, emitter: CodeEmitter): void {
  // TFLite SPLIT_V: input(0) = data, input(1) = size_splits, input(2) = axis
  const data = emitter.ref(node.inputs[0]);

  emitter.comment(`SPLIT_V into ${node.outputs.length} parts`);

  for (let i = 0; i < node.outputs.length; i++) {
    const out = emitter.declare(node.outputs[i]);
    emitter.line(`const ${out} = builder.split(${data}, ${node.outputs.length})[${i}]; // SPLIT_V`);
  }
}

function emitSlice(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // TFLite SLICE: inputs are data, begin, size
  if (node.inputs.length >= 3) {
    const begin = emitter.ref(node.inputs[1]);
    const size = emitter.ref(node.inputs[2]);
    emitter.line(`const ${output} = builder.slice(${input}, ${begin}, ${size});`);
  } else {
    emitter.line(`const ${output} = ${input}; // SLICE (missing begin/size)`);
  }
}

function emitStridedSlice(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  emitter.comment('STRIDED_SLICE');
  emitter.line(`const ${output} = builder.slice(${input}); // STRIDED_SLICE — simplified`);
}

function emitPad(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // TFLite PAD/PADV2: inputs[1] is a constant [N, 2] int32 tensor
  // WebNN pad() requires (input, beginningPadding, endingPadding, options?)
  if (node.inputs.length >= 2 && emitter.isConstant(node.inputs[1])) {
    const rawData = emitter.constantRawData(node.inputs[1]);
    if (rawData) {
      const int32View = new Int32Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 4);
      const n = int32View.length / 2;
      const beginning: number[] = [];
      const ending: number[] = [];
      for (let i = 0; i < n; i++) {
        beginning.push(int32View[i * 2]);
        ending.push(int32View[i * 2 + 1]);
      }
      if (node.inputs.length >= 3 && node.inputs[2] !== '') {
        const constVal = emitter.ref(node.inputs[2]);
        emitter.line(`const ${output} = builder.pad(${input}, ${JSON.stringify(beginning)}, ${JSON.stringify(ending)}, { value: ${constVal} });`);
      } else {
        emitter.line(`const ${output} = builder.pad(${input}, ${JSON.stringify(beginning)}, ${JSON.stringify(ending)});`);
      }
      return;
    }
  }
  // Fallback: emit with empty padding arrays
  emitter.comment('WARNING: pad constant data not available');
  emitter.line(`const ${output} = builder.pad(${input}, [], []);`);
}

function emitTile(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  if (node.inputs.length >= 2) {
    const multiples = emitter.ref(node.inputs[1]);
    emitter.line(`const ${output} = builder.tile(${input}, ${multiples});`);
  } else {
    emitter.line(`const ${output} = ${input}; // TILE (missing multiples)`);
  }
}

function emitGather(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  const axis = (node.attributes.axis as number) ?? 0;

  emitter.line(`const ${output} = builder.gather(${input}, ${indices}, { axis: ${axis} });`);
}

function emitGatherND(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);

  emitter.line(`const ${output} = builder.gatherND(${input}, ${indices});`);
}

function emitScatterND(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const updates = emitter.ref(node.inputs[2]);
  const output = emitter.declare(node.outputs[0]);

  emitter.line(`const ${output} = builder.scatterND(${input}, ${indices}, ${updates});`);
}

function emitShape(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.shape(${input});`);
}

function emitPack(node: NodeIR, emitter: CodeEmitter): void {
  const inputs = node.inputs.map((n) => emitter.ref(n));
  const output = emitter.declare(node.outputs[0]);
  const axis = (node.attributes.axis as number) ?? 0;

  emitter.comment(`PACK along axis ${axis}`);
  // Pack = stack = unsqueeze each input + concat
  const unsqueezed = inputs.map((inp, i) => {
    const uName = `${output}_u${i}`;
    emitter.line(`const ${uName} = builder.unsqueeze(${inp}, { axes: [${axis}] });`);
    return uName;
  });
  emitter.line(`const ${output} = builder.concat([${unsqueezed.join(', ')}], ${axis});`);
}

function emitUnpack(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const axis = (node.attributes.axis as number) ?? 0;
  const num = (node.attributes.num as number) ?? node.outputs.length;

  emitter.comment(`UNPACK along axis ${axis}`);
  for (let i = 0; i < node.outputs.length; i++) {
    const out = emitter.declare(node.outputs[i]);
    emitter.line(`const ${out} = builder.squeeze(builder.split(${input}, ${num}, { axis: ${axis} })[${i}], { axes: [${axis}] });`);
  }
}

// ──────────────────────────────────────────────────────
// Fully connected (dense) op
// ──────────────────────────────────────────────────────

function emitFullyConnected(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const weights = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  const bias = node.inputs.length > 2 ? emitter.ref(node.inputs[2]) : undefined;

  emitter.comment('FULLY_CONNECTED → gemm');

  // TFLite FullyConnected does: output = input * weights^T + bias
  const rawVar = `${output}_raw`;
  if (bias) {
    emitter.line(`const ${rawVar} = builder.gemm(${input}, ${weights}, { c: ${bias}, bTranspose: true });`);
  } else {
    emitter.line(`const ${rawVar} = builder.gemm(${input}, ${weights}, { bTranspose: true });`);
  }

  const activation = node.attributes.fused_activation as string | undefined;
  const resultVar = emitFusedActivation(rawVar, activation, emitter);
  if (resultVar !== output) {
    emitter.line(`const ${output} = ${resultVar};`);
  }
}

// ──────────────────────────────────────────────────────
// Reduce ops
// ──────────────────────────────────────────────────────

function emitReduce(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const opMap: Record<string, string> = {
    MEAN: 'reduceMean',
    SUM: 'reduceSum',
    REDUCE_PROD: 'reduceProduct',
    REDUCE_MAX: 'reduceMax',
    REDUCE_MIN: 'reduceMin',
    REDUCE_ANY: 'reduceSum',   // approximate — no direct WebNN equivalent for bool reduce
    REDUCE_ALL: 'reduceProduct', // approximate
  };

  const webnnOp = opMap[node.opType] ?? 'reduceMean';
  const keepDims = node.attributes.keep_dims as boolean ?? false;

  // Axes come from the second input tensor
  if (node.inputs.length > 1) {
    const axes = emitter.ref(node.inputs[1]);
    emitter.line(`const ${output} = builder.${webnnOp}(${input}, { axes: ${axes}, keepDimensions: ${keepDims} });`);
  } else {
    emitter.line(`const ${output} = builder.${webnnOp}(${input}, { keepDimensions: ${keepDims} });`);
  }
}

// ──────────────────────────────────────────────────────
// Softmax
// ──────────────────────────────────────────────────────

function emitSoftmax(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // TFLite softmax always operates on last axis
  emitter.line(`const ${output} = builder.softmax(${input});`);
}

// ──────────────────────────────────────────────────────
// Resize ops
// ──────────────────────────────────────────────────────

function emitResizeBilinear(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const opts: string[] = [`mode: 'linear'`];
  if (node.attributes.half_pixel_centers) {
    opts.push(`coordinateTransformMode: 'half_pixel'`);
  } else if (node.attributes.align_corners) {
    opts.push(`coordinateTransformMode: 'align_corners'`);
  }
  // TFLite uses NHWC — spatial dims are axes 1 and 2
  opts.push(`axes: [1, 2]`);

  // Second input is a constant [height, width] int32 tensor — extract as JS array
  if (node.inputs.length > 1 && emitter.isConstant(node.inputs[1])) {
    const rawData = emitter.constantRawData(node.inputs[1]);
    if (rawData) {
      const int32View = new Int32Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 4);
      opts.push(`sizes: [${int32View[0]}, ${int32View[1]}]`);
    }
  }

  emitter.comment('RESIZE_BILINEAR');
  emitter.line(`const ${output} = builder.resample2d(${input}, { ${opts.join(', ')} });`);
}

function emitResizeNearestNeighbor(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const opts: string[] = [`mode: 'nearest-neighbor'`];
  if (node.attributes.half_pixel_centers) {
    opts.push(`coordinateTransformMode: 'half_pixel'`);
  } else if (node.attributes.align_corners) {
    opts.push(`coordinateTransformMode: 'align_corners'`);
  }
  // TFLite uses NHWC — spatial dims are axes 1 and 2
  opts.push(`axes: [1, 2]`);

  // Second input is a constant [height, width] int32 tensor — extract as JS array
  if (node.inputs.length > 1 && emitter.isConstant(node.inputs[1])) {
    const rawData = emitter.constantRawData(node.inputs[1]);
    if (rawData) {
      const int32View = new Int32Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 4);
      opts.push(`sizes: [${int32View[0]}, ${int32View[1]}]`);
    }
  }

  emitter.comment('RESIZE_NEAREST_NEIGHBOR');
  emitter.line(`const ${output} = builder.resample2d(${input}, { ${opts.join(', ')} });`);
}

// ──────────────────────────────────────────────────────
// Other ops
// ──────────────────────────────────────────────────────

function emitCast(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const outType = (node.attributes.out_data_type as string) ?? 'FLOAT32';
  emitter.line(`const ${output} = builder.cast(${input}, '${outType.toLowerCase()}');`);
}

function emitArgMax(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // Axis from second input
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
  // block_size from attributes or second input
  emitter.line(`const ${output} = builder.depthToSpace(${input});`);
}

function emitSpaceToDepth(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.spaceToDepth(${input});`);
}

function emitBatchMatMul(node: NodeIR, emitter: CodeEmitter): void {
  const a = emitter.ref(node.inputs[0]);
  const b = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.matmul(${a}, ${b});`);
}

function emitCumSum(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.cumulativeSum(${input}, 0);`);
}

// TFLite DEQUANTIZE — quantization params are in tensor metadata (stored as attributes by parser)
// WebNN signature: builder.dequantizeLinear(input, scale, zeroPoint)
function emitDequantize(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.comment('DEQUANTIZE');

  // Find the input tensor's quantization params from attributes
  // Parser stores them as input_<tensorIndex>_scale and input_<tensorIndex>_zero_point
  const scaleValues = findQuantizationScale(node);
  const zpValues = findQuantizationZeroPoint(node);

  if (scaleValues && scaleValues.length > 0) {
    const scaleVar = `${output}_scale`;
    if (scaleValues.length === 1) {
      emitter.line(`const ${scaleVar} = builder.constant({dataType: 'float32', shape: []}, new Float32Array([${scaleValues[0]}]));`);
    } else {
      emitter.line(`const ${scaleVar} = builder.constant({dataType: 'float32', shape: [${scaleValues.length}]}, new Float32Array([${scaleValues.join(', ')}]));`);
    }

    const zpVar = `${output}_zp`;
    if (zpValues && zpValues.length > 0) {
      if (zpValues.length === 1) {
        emitter.line(`const ${zpVar} = builder.constant({dataType: 'int32', shape: []}, new Int32Array([${zpValues[0]}]));`);
      } else {
        emitter.line(`const ${zpVar} = builder.constant({dataType: 'int32', shape: [${zpValues.length}]}, new Int32Array([${zpValues.join(', ')}]));`);
      }
    } else {
      emitter.line(`const ${zpVar} = builder.constant({dataType: 'int32', shape: []}, new Int32Array([0]));`);
    }

    emitter.line(`const ${output} = builder.dequantizeLinear(${input}, ${scaleVar}, ${zpVar});`);
  } else {
    // No quantization params found — emit identity (cast to float32)
    emitter.comment('No quantization params found — using cast as fallback');
    emitter.line(`const ${output} = builder.cast(${input}, 'float32');`);
  }
}

// TFLite QUANTIZE — quantization params are in output tensor metadata
function emitQuantize(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.comment('QUANTIZE');

  const scaleValues = findQuantizationScale(node);
  const zpValues = findQuantizationZeroPoint(node);

  if (scaleValues && scaleValues.length > 0) {
    const scaleVar = `${output}_scale`;
    if (scaleValues.length === 1) {
      emitter.line(`const ${scaleVar} = builder.constant({dataType: 'float32', shape: []}, new Float32Array([${scaleValues[0]}]));`);
    } else {
      emitter.line(`const ${scaleVar} = builder.constant({dataType: 'float32', shape: [${scaleValues.length}]}, new Float32Array([${scaleValues.join(', ')}]));`);
    }

    const zpVar = `${output}_zp`;
    if (zpValues && zpValues.length > 0) {
      if (zpValues.length === 1) {
        emitter.line(`const ${zpVar} = builder.constant({dataType: 'uint8', shape: []}, new Uint8Array([${zpValues[0]}]));`);
      } else {
        emitter.line(`const ${zpVar} = builder.constant({dataType: 'uint8', shape: [${zpValues.length}]}, new Uint8Array([${zpValues.join(', ')}]));`);
      }
    } else {
      emitter.line(`const ${zpVar} = builder.constant({dataType: 'uint8', shape: []}, new Uint8Array([0]));`);
    }

    emitter.line(`const ${output} = builder.quantizeLinear(${input}, ${scaleVar}, ${zpVar});`);
  } else {
    // No quantization params found — emit identity
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

function emitL2Normalization(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.comment('L2_NORMALIZATION');
  // L2 norm: x / sqrt(sum(x^2))
  const squared = `${output}_sq`;
  const sumSq = `${output}_sum`;
  const sqrtSumSq = `${output}_sqrt`;
  emitter.line(`const ${squared} = builder.mul(${input}, ${input});`);
  emitter.line(`const ${sumSq} = builder.reduceSum(${squared}, { keepDimensions: true });`);
  emitter.line(`const ${sqrtSumSq} = builder.sqrt(${sumSq});`);
  emitter.line(`const ${output} = builder.div(${input}, ${sqrtSumSq});`);
}

function emitLocalResponseNormalization(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.comment('LOCAL_RESPONSE_NORMALIZATION');
  emitter.line(`const ${output} = ${input}; // LRN — requires custom decomposition`);
}

function emitLogSoftmax(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.log(builder.softmax(${input}));`);
}

function emitMirrorPad(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // TFLite MIRROR_PAD: inputs[1] is a constant [N, 2] int32 tensor
  // WebNN pad() requires (input, beginningPadding, endingPadding, options?)
  if (node.inputs.length > 1 && emitter.isConstant(node.inputs[1])) {
    const rawData = emitter.constantRawData(node.inputs[1]);
    if (rawData) {
      const int32View = new Int32Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 4);
      const n = int32View.length / 2;
      const beginning: number[] = [];
      const ending: number[] = [];
      for (let i = 0; i < n; i++) {
        beginning.push(int32View[i * 2]);
        ending.push(int32View[i * 2 + 1]);
      }
      emitter.line(`const ${output} = builder.pad(${input}, ${JSON.stringify(beginning)}, ${JSON.stringify(ending)}, { mode: 'reflection' });`);
      return;
    }
  }
  // Fallback
  emitter.comment('WARNING: mirror_pad constant data not available');
  emitter.line(`const ${output} = builder.pad(${input}, [], [], { mode: 'reflection' });`);
}

// ──────────────────────────────────────────────────────
// Register all TFLite ops
// ──────────────────────────────────────────────────────

// Convolution
registerTfliteOp('CONV_2D', emitConv2D);
registerTfliteOp('DEPTHWISE_CONV_2D', emitDepthwiseConv2D);
registerTfliteOp('TRANSPOSE_CONV', emitTransposeConv);

// Pooling
registerTfliteOp('AVERAGE_POOL_2D', emitPool2D);
registerTfliteOp('MAX_POOL_2D', emitPool2D);
registerTfliteOp('L2_POOL_2D', emitL2Pool2D);

// Activations
registerTfliteOp('RELU', emitSimpleActivation);
registerTfliteOp('RELU6', emitSimpleActivation);
registerTfliteOp('RELU_N1_TO_1', emitSimpleActivation);
registerTfliteOp('RELU_0_TO_1', emitSimpleActivation);
registerTfliteOp('LOGISTIC', emitSimpleActivation);
registerTfliteOp('TANH', emitSimpleActivation);
registerTfliteOp('ELU', emitSimpleActivation);
registerTfliteOp('HARD_SWISH', emitSimpleActivation);
registerTfliteOp('LEAKY_RELU', emitSimpleActivation);
registerTfliteOp('GELU', emitSimpleActivation);
registerTfliteOp('PRELU', emitSimpleActivation);

// Binary
registerTfliteOp('ADD', emitBinaryOp);
registerTfliteOp('SUB', emitBinaryOp);
registerTfliteOp('MUL', emitBinaryOp);
registerTfliteOp('DIV', emitBinaryOp);
registerTfliteOp('POW', emitBinaryOp);
registerTfliteOp('MAXIMUM', emitBinaryOp);
registerTfliteOp('MINIMUM', emitBinaryOp);
registerTfliteOp('FLOOR_DIV', emitBinaryOp);

// Unary
registerTfliteOp('ABS', emitUnaryOp);
registerTfliteOp('CEIL', emitUnaryOp);
registerTfliteOp('COS', emitUnaryOp);
registerTfliteOp('EXP', emitUnaryOp);
registerTfliteOp('FLOOR', emitUnaryOp);
registerTfliteOp('LOG', emitUnaryOp);
registerTfliteOp('NEG', emitUnaryOp);
registerTfliteOp('SIN', emitUnaryOp);
registerTfliteOp('SQRT', emitUnaryOp);
registerTfliteOp('RSQRT', emitUnaryOp);
registerTfliteOp('ROUND', emitUnaryOp);
registerTfliteOp('SIGN', emitUnaryOp);
registerTfliteOp('SQUARE', emitUnaryOp);

// Logical / comparison
registerTfliteOp('EQUAL', emitLogicalOp);
registerTfliteOp('NOT_EQUAL', emitLogicalOp);
registerTfliteOp('GREATER', emitLogicalOp);
registerTfliteOp('GREATER_EQUAL', emitLogicalOp);
registerTfliteOp('LESS', emitLogicalOp);
registerTfliteOp('LESS_EQUAL', emitLogicalOp);
registerTfliteOp('LOGICAL_AND', emitLogicalOp);
registerTfliteOp('LOGICAL_OR', emitLogicalOp);
registerTfliteOp('LOGICAL_NOT', emitLogicalOp);

// Shape / manipulation
registerTfliteOp('RESHAPE', emitReshape);
registerTfliteOp('SQUEEZE', emitSqueeze);
registerTfliteOp('EXPAND_DIMS', emitExpandDims);
registerTfliteOp('TRANSPOSE', emitTranspose);
registerTfliteOp('CONCATENATION', emitConcatenation);
registerTfliteOp('SPLIT', emitSplit);
registerTfliteOp('SPLIT_V', emitSplitV);
registerTfliteOp('SLICE', emitSlice);
registerTfliteOp('STRIDED_SLICE', emitStridedSlice);
registerTfliteOp('PAD', emitPad);
registerTfliteOp('PADV2', emitPad);
registerTfliteOp('MIRROR_PAD', emitMirrorPad);
registerTfliteOp('TILE', emitTile);
registerTfliteOp('GATHER', emitGather);
registerTfliteOp('GATHER_ND', emitGatherND);
registerTfliteOp('SCATTER_ND', emitScatterND);
registerTfliteOp('SHAPE', emitShape);
registerTfliteOp('PACK', emitPack);
registerTfliteOp('UNPACK', emitUnpack);

// Dense / matmul
registerTfliteOp('FULLY_CONNECTED', emitFullyConnected);
registerTfliteOp('BATCH_MATMUL', emitBatchMatMul);

// Reduce
registerTfliteOp('MEAN', emitReduce);
registerTfliteOp('SUM', emitReduce);
registerTfliteOp('REDUCE_PROD', emitReduce);
registerTfliteOp('REDUCE_MAX', emitReduce);
registerTfliteOp('REDUCE_MIN', emitReduce);
registerTfliteOp('REDUCE_ANY', emitReduce);
registerTfliteOp('REDUCE_ALL', emitReduce);

// Softmax
registerTfliteOp('SOFTMAX', emitSoftmax);
registerTfliteOp('LOG_SOFTMAX', emitLogSoftmax);

// Resize
registerTfliteOp('RESIZE_BILINEAR', emitResizeBilinear);
registerTfliteOp('RESIZE_NEAREST_NEIGHBOR', emitResizeNearestNeighbor);

// Other
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
registerTfliteOp('L2_NORMALIZATION', emitL2Normalization);
registerTfliteOp('LOCAL_RESPONSE_NORMALIZATION', emitLocalResponseNormalization);
