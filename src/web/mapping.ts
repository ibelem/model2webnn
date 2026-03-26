// Op Mapping module — side-by-side view of source model ops → WebNN ops
// Shows inputs/outputs, data types for each node.

import type { ConvertResult } from '../index.js';
import { getEmitter } from '../operators/registry.js';
import { toJsVarName } from '../ir/graph.js';

// Known ONNX → WebNN op name mappings (for display purposes)
const onnxToWebnn: Record<string, string> = {
  // Activations
  Relu: 'relu', LeakyRelu: 'leakyRelu', Sigmoid: 'sigmoid', Tanh: 'tanh',
  Elu: 'elu', Softmax: 'softmax', LogSoftmax: 'softmax + log', HardSigmoid: 'hardSigmoid',
  HardSwish: 'hardSwish', Gelu: 'gelu', Selu: 'elu', Softsign: 'softsign', Softplus: 'softplus',
  // Binary / element-wise
  Add: 'add', Sub: 'sub', Mul: 'mul', Div: 'div', Pow: 'pow',
  Min: 'min', Max: 'max', Mean: 'add + div (mean)', PRelu: 'prelu',
  // Unary
  Abs: 'abs', Neg: 'neg', Ceil: 'ceil', Floor: 'floor', Exp: 'exp', Log: 'log',
  Sqrt: 'sqrt', Reciprocal: 'reciprocal', Erf: 'erf', Identity: 'identity',
  Cast: 'cast', Sin: 'sin', Cos: 'cos', Tan: 'tan', Not: 'logicalNot', Sign: 'sign',
  Round: 'roundEven (custom)',
  IsInf: 'isInf (custom)', IsNaN: 'isNaN (custom)',
  // Logical
  Equal: 'equal', Greater: 'greater', GreaterOrEqual: 'greaterOrEqual',
  Less: 'less', LessOrEqual: 'lessOrEqual',
  And: 'logicalAnd', Or: 'logicalOr', Xor: 'logicalXor',
  // Conv
  Conv: 'conv2d', ConvInteger: 'conv2d (integer)', ConvTranspose: 'convTranspose2d',
  // Gemm / MatMul
  Gemm: 'gemm', MatMul: 'matmul', MatMulInteger: 'matmulInteger',
  // Normalization
  BatchNormalization: 'batchNormalization', LayerNormalization: 'layerNormalization',
  InstanceNormalization: 'instanceNormalization', GroupNormalization: 'layerNormalization',
  SimplifiedLayerNormalization: 'layerNormalization', SkipSimplifiedLayerNormalization: 'layerNormalization + add',
  LRN: 'localResponseNormalization',
  // Pool
  AveragePool: 'averagePool2d', MaxPool: 'maxPool2d',
  GlobalAveragePool: 'averagePool2d (global)', GlobalMaxPool: 'maxPool2d (global)',
  LpPool: 'l2Pool2d', GlobalLpPool: 'l2Pool2d (global)',
  // Reduce
  ReduceMean: 'reduceMean', ReduceSum: 'reduceSum', ReduceMax: 'reduceMax',
  ReduceMin: 'reduceMin', ReduceL2: 'reduceL2', ReduceL1: 'reduceL1',
  ReduceProd: 'reduceProduct', ReduceLogSumExp: 'reduceLogSumExp',
  ReduceSumSquare: 'reduceSumSquare', ReduceLogSum: 'reduceLogSum',
  // Shape / layout
  Reshape: 'reshape', Transpose: 'transpose', Concat: 'concat', Split: 'split',
  Flatten: 'reshape', Squeeze: 'reshape', Unsqueeze: 'reshape',
  Pad: 'pad', Slice: 'slice', Gather: 'gather', Where: 'where',
  Tile: 'expand', Expand: 'expand', Clip: 'clamp',
  Resize: 'resample2d', Shape: 'constant (shape)',
  ArgMax: 'argMax', ArgMin: 'argMin',
  GatherElements: 'gatherElements', GatherND: 'gatherND',
  ScatterElements: 'scatterElements', ScatterND: 'scatterND',
  CumSum: 'cumulativeSum', DepthToSpace: 'depthToSpace', SpaceToDepth: 'spaceToDepth',
  Trilu: 'triangular',
  Dropout: 'identity (dropout)', // pass-through at inference
  // Quantization
  DequantizeLinear: 'dequantizeLinear', QuantizeLinear: 'quantizeLinear',
  DynamicQuantizeLinear: 'dequantizeLinear (dynamic)',
  GatherBlockQuantized: 'gather + dequantize',
  // Composite
  GroupQueryAttention: 'GQA (composite)', MultiHeadAttention: 'MHA (composite)',
  RotaryEmbedding: 'RotaryEmbedding (composite)', MatMulNBits: 'matmul (dequantized)',
  LSTM: 'lstm (composite)', GRU: 'gru (composite)', Einsum: 'einsum (composite)',
};

const tfliteToWebnn: Record<string, string> = {
  // Conv
  CONV_2D: 'conv2d', DEPTHWISE_CONV_2D: 'conv2d (depthwise)', TRANSPOSE_CONV: 'convTranspose2d',
  // Pool
  AVERAGE_POOL_2D: 'averagePool2d', MAX_POOL_2D: 'maxPool2d', L2_POOL_2D: 'l2Pool2d',
  // Binary / element-wise
  ADD: 'add', MUL: 'mul', SUB: 'sub', DIV: 'div', POW: 'pow',
  MAXIMUM: 'max', MINIMUM: 'min', FLOOR_DIV: 'div + floor',
  ADD_N: 'add (n-way)',
  // Activation
  RELU: 'relu', RELU6: 'clamp', RELU_N1_TO_1: 'clamp', RELU_0_TO_1: 'clamp',
  LOGISTIC: 'sigmoid', TANH: 'tanh', ELU: 'elu', HARD_SWISH: 'hardSwish',
  LEAKY_RELU: 'leakyRelu', GELU: 'gelu', PRELU: 'prelu',
  // Unary
  ABS: 'abs', CEIL: 'ceil', COS: 'cos', EXP: 'exp', FLOOR: 'floor',
  LOG: 'log', NEG: 'neg', SIN: 'sin', SQRT: 'sqrt', RSQRT: 'reciprocal(sqrt)',
  ROUND: 'roundEven', SIGN: 'sign', SQUARE: 'mul (x*x)',
  // Logical
  EQUAL: 'equal', NOT_EQUAL: 'notEqual (custom)', GREATER: 'greater',
  GREATER_EQUAL: 'greaterOrEqual', LESS: 'less', LESS_EQUAL: 'lessOrEqual',
  LOGICAL_AND: 'logicalAnd', LOGICAL_OR: 'logicalOr', LOGICAL_NOT: 'logicalNot',
  // Shape / layout
  RESHAPE: 'reshape', SQUEEZE: 'reshape', EXPAND_DIMS: 'reshape',
  TRANSPOSE: 'transpose', CONCATENATION: 'concat',
  SPLIT: 'split', SPLIT_V: 'split',
  SLICE: 'slice', STRIDED_SLICE: 'slice',
  PAD: 'pad', PADV2: 'pad', MIRROR_PAD: 'pad (reflect)',
  TILE: 'expand', GATHER: 'gather', GATHER_ND: 'gatherND', SCATTER_ND: 'scatterND',
  SHAPE: 'constant (shape)', PACK: 'concat + reshape', UNPACK: 'split + reshape',
  BROADCAST_TO: 'expand',
  // Reduction
  MEAN: 'reduceMean', SUM: 'reduceSum', REDUCE_PROD: 'reduceProduct',
  REDUCE_MAX: 'reduceMax', REDUCE_MIN: 'reduceMin',
  REDUCE_ANY: 'reduceMax (bool)', REDUCE_ALL: 'reduceMin (bool)',
  // Normalization
  SOFTMAX: 'softmax', LOG_SOFTMAX: 'softmax + log',
  L2_NORMALIZATION: 'l2Normalization', LOCAL_RESPONSE_NORMALIZATION: 'localResponseNormalization',
  // Dense / MatMul
  FULLY_CONNECTED: 'gemm', BATCH_MATMUL: 'matmul',
  // Resize
  RESIZE_BILINEAR: 'resample2d', RESIZE_NEAREST_NEIGHBOR: 'resample2d (nearest)',
  // Misc
  CAST: 'cast', ARG_MAX: 'argMax', ARG_MIN: 'argMin',
  DEPTH_TO_SPACE: 'depthToSpace', SPACE_TO_DEPTH: 'spaceToDepth',
  CUMSUM: 'cumulativeSum',
  DEQUANTIZE: 'dequantizeLinear', QUANTIZE: 'quantizeLinear',
  WHERE: 'where', SELECT: 'where', SELECT_V2: 'where',
  ZEROS_LIKE: 'constant (zeros)',
};

export function initMapping(result: ConvertResult): void {
  const tbody = document.getElementById('mappingBody')!;
  tbody.innerHTML = '';

  const graph = result.graph;

  // Build a tensor info map for data types
  const tensorTypes = new Map<string, string>();
  for (const t of graph.inputs) tensorTypes.set(t.name, t.dataType);
  for (const t of graph.outputs) tensorTypes.set(t.name, t.dataType);
  for (const c of graph.constants) tensorTypes.set(c.name, c.dataType);

  const fragment = document.createDocumentFragment();

  for (const node of graph.nodes) {
    // Propagate output types FIRST so outputTypes lookup below can find them
    for (let i = 0; i < node.outputs.length; i++) {
      if (node.outputs[i] && !tensorTypes.has(node.outputs[i])) {
        const firstInputType = node.inputs.find((n) => n !== '' && tensorTypes.has(n));
        if (firstInputType) tensorTypes.set(node.outputs[i], tensorTypes.get(firstInputType)!);
      }
    }

    const row = document.createElement('tr');
    const supported = !!getEmitter(graph.format, node.opType);
    const webnnOp = getWebnnOpName(graph.format, node.opType);

    const inputNames = node.inputs.filter((n) => n !== '').map((n) => truncate(n, 30));
    const outputNames = node.outputs.filter((n) => n !== '').map((n) => truncate(n, 30));
    const webnnInputNames = node.inputs.filter((n) => n !== '').map((n) => truncate(toJsVarName(n), 30));
    const webnnOutputNames = node.outputs.filter((n) => n !== '').map((n) => truncate(toJsVarName(n), 30));

    const inputTypes = node.inputs.filter((n) => n !== '').map((n) => tensorTypes.get(n) ?? '?');
    const outputTypes = node.outputs.filter((n) => n !== '').map((n) => tensorTypes.get(n) ?? '?');

    const webnnOpClass = supported ? 'mapping-webnn-op' : 'mapping-unsupported';
    const webnnOpText = supported ? webnnOp : `\u26A0 ${node.opType} (unsupported)`;

    row.innerHTML = `
      <td><span class="mapping-op-name">${escapeHtml(node.opType)}</span></td>
      <td class="mapping-tensors" title="${escapeHtml(node.inputs.join(', '))}">${escapeHtml(inputNames.join(', '))}</td>
      <td class="mapping-tensors" title="${escapeHtml(node.outputs.join(', '))}">${escapeHtml(outputNames.join(', '))}</td>
      <td>${escapeHtml(uniqueTypes(inputTypes))}</td>
      <td><span class="${webnnOpClass}">${escapeHtml(webnnOpText)}</span></td>
      <td class="mapping-tensors" title="${escapeHtml(node.inputs.map(toJsVarName).join(', '))}">${escapeHtml(webnnInputNames.join(', '))}</td>
      <td class="mapping-tensors" title="${escapeHtml(node.outputs.map(toJsVarName).join(', '))}">${escapeHtml(webnnOutputNames.join(', '))}</td>
      <td>${escapeHtml(uniqueTypes(outputTypes))}</td>
    `;

    fragment.appendChild(row);
  }

  tbody.appendChild(fragment);
}

function getWebnnOpName(format: string, opType: string): string {
  if (format === 'onnx') return onnxToWebnn[opType] ?? opType.toLowerCase();
  if (format === 'tflite') return tfliteToWebnn[opType] ?? opType.toLowerCase();
  return opType;
}

function uniqueTypes(types: string[]): string {
  return [...new Set(types)].join(', ');
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
