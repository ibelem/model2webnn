// TFLite FlatBuffers parser → GraphIR
// Parses .tflite files and produces a format-agnostic GraphIR.
// Uses a manual FlatBuffers reader to avoid schema generation dependencies.
//
// Schema reference (LiteRT, successor to TensorFlow Lite):
//   https://github.com/google-ai-edge/LiteRT/blob/main/tflite/converter/schema/schema.fbs

import type { GraphIR, TensorInfo, ConstantInfo, NodeIR, MLOperandDataType } from '../ir/graph.js';
import { tfliteDataType } from '../ir/graph.js';

// TFLite TensorType enum values
// See: https://github.com/google-ai-edge/LiteRT/blob/main/tflite/converter/schema/schema.fbs
const TensorType = {
  FLOAT32: 0,
  FLOAT16: 1,
  INT32: 2,
  UINT8: 3,
  INT64: 4,
  STRING: 5,
  BOOL: 6,
  INT16: 7,
  COMPLEX64: 8,
  INT8: 9,
  FLOAT64: 10,
  COMPLEX128: 11,
  UINT64: 12,
  RESOURCE: 13,
  VARIANT: 14,
  UINT32: 15,
  UINT16: 16,
  INT4: 17,
  BFLOAT16: 18,
} as const;

// TFLite BuiltinOperator enum values — commonly used ops
// See: https://github.com/google-ai-edge/LiteRT/blob/main/tflite/converter/schema/schema.fbs
const BuiltinOperator: Record<number, string> = {
  0: 'ADD',
  1: 'AVERAGE_POOL_2D',
  2: 'CONCATENATION',
  3: 'CONV_2D',
  4: 'DEPTHWISE_CONV_2D',
  5: 'DEPTH_TO_SPACE',
  6: 'DEQUANTIZE',
  7: 'EMBEDDING_LOOKUP',
  8: 'FLOOR',
  9: 'FULLY_CONNECTED',
  10: 'HASHTABLE_LOOKUP',
  11: 'L2_NORMALIZATION',
  12: 'L2_POOL_2D',
  13: 'LOCAL_RESPONSE_NORMALIZATION',
  14: 'LOGISTIC',
  15: 'LSH_PROJECTION',
  16: 'LSTM',
  17: 'MAX_POOL_2D',
  18: 'MUL',
  19: 'RELU',
  20: 'RELU_N1_TO_1',
  21: 'RELU6',
  22: 'RESHAPE',
  23: 'RESIZE_BILINEAR',
  25: 'SOFTMAX',
  26: 'SPACE_TO_DEPTH',
  27: 'SVDF',
  28: 'TANH',
  29: 'CONCAT_EMBEDDINGS',
  30: 'SKIP_GRAM',
  31: 'CALL',
  32: 'CUSTOM',
  33: 'EMBEDDING_LOOKUP_SPARSE',
  34: 'PAD',
  35: 'UNIDIRECTIONAL_SEQUENCE_RNN',
  36: 'GATHER',
  37: 'BATCH_TO_SPACE_ND',
  38: 'SPACE_TO_BATCH_ND',
  39: 'TRANSPOSE',
  40: 'MEAN',
  41: 'SUB',
  42: 'DIV',
  43: 'SQUEEZE',
  44: 'UNIDIRECTIONAL_SEQUENCE_LSTM',
  45: 'STRIDED_SLICE',
  46: 'BIDIRECTIONAL_SEQUENCE_RNN',
  47: 'EXP',
  48: 'TOPK_V2',
  49: 'SPLIT',
  50: 'LOG_SOFTMAX',
  51: 'DELEGATE',
  52: 'BIDIRECTIONAL_SEQUENCE_LSTM',
  53: 'CAST',
  54: 'PRELU',
  55: 'MAXIMUM',
  56: 'ARG_MAX',
  57: 'MINIMUM',
  58: 'LESS',
  59: 'NEG',
  60: 'PADV2',
  61: 'GREATER',
  62: 'GREATER_EQUAL',
  63: 'LESS_EQUAL',
  64: 'SELECT',
  65: 'SLICE',
  66: 'SIN',
  67: 'TRANSPOSE_CONV',
  68: 'SPARSE_TO_DENSE',
  69: 'TILE',
  70: 'EXPAND_DIMS',
  71: 'EQUAL',
  72: 'NOT_EQUAL',
  73: 'LOG',
  74: 'SUM',
  75: 'SQRT',
  76: 'RSQRT',
  77: 'SHAPE',
  78: 'POW',
  79: 'ARG_MIN',
  80: 'FAKE_QUANT',
  81: 'REDUCE_PROD',
  82: 'REDUCE_MAX',
  83: 'PACK',
  84: 'LOGICAL_OR',
  85: 'ONE_HOT',
  86: 'LOGICAL_AND',
  87: 'LOGICAL_NOT',
  88: 'UNPACK',
  89: 'REDUCE_MIN',
  90: 'FLOOR_DIV',
  91: 'REDUCE_ANY',
  92: 'SQUARE',
  93: 'ZEROS_LIKE',
  94: 'FILL',
  95: 'FLOOR_MOD',
  96: 'RANGE',
  97: 'RESIZE_NEAREST_NEIGHBOR',
  98: 'LEAKY_RELU',
  99: 'SQUARED_DIFFERENCE',
  100: 'MIRROR_PAD',
  101: 'ABS',
  102: 'SPLIT_V',
  103: 'UNIQUE',
  104: 'CEIL',
  105: 'REVERSE_V2',
  106: 'ADD_N',
  107: 'GATHER_ND',
  108: 'COS',
  109: 'WHERE',
  110: 'RANK',
  111: 'ELU',
  112: 'REVERSE_SEQUENCE',
  113: 'MATRIX_DIAG',
  114: 'QUANTIZE',
  115: 'MATRIX_SET_DIAG',
  116: 'ROUND',
  117: 'HARD_SWISH',
  118: 'IF',
  119: 'WHILE',
  120: 'NON_MAX_SUPPRESSION_V4',
  121: 'NON_MAX_SUPPRESSION_V5',
  122: 'SCATTER_ND',
  123: 'SELECT_V2',
  124: 'DENSIFY',
  125: 'SEGMENT_SUM',
  126: 'BATCH_MATMUL',
  127: 'PLACEHOLDER_FOR_GREATER_OP_CODES',
  128: 'CUMSUM',
  129: 'CALL_ONCE',
  130: 'BROADCAST_TO',
  131: 'RFFT2D',
  132: 'CONV_3D',
  133: 'IMAG',
  134: 'REAL',
  135: 'COMPLEX_ABS',
  136: 'HASHTABLE',
  137: 'HASHTABLE_FIND',
  138: 'HASHTABLE_IMPORT',
  139: 'HASHTABLE_SIZE',
  140: 'REDUCE_ALL',
  141: 'CONV_3D_TRANSPOSE',
  142: 'VAR_HANDLE',
  143: 'READ_VARIABLE',
  144: 'ASSIGN_VARIABLE',
  145: 'BROADCAST_ARGS',
  146: 'RANDOM_STANDARD_NORMAL',
  147: 'BUCKETIZE',
  148: 'RANDOM_UNIFORM',
  149: 'MULTINOMIAL',
  150: 'GELU',
  151: 'DYNAMIC_UPDATE_SLICE',
  152: 'RELU_0_TO_1',
  153: 'UNSORTED_SEGMENT_PROD',
  154: 'UNSORTED_SEGMENT_MAX',
  155: 'UNSORTED_SEGMENT_SUM',
  156: 'ATAN2',
  157: 'UNSORTED_SEGMENT_MIN',
  158: 'SIGN',
  159: 'BITCAST',
  160: 'BITWISE_XOR',
  161: 'RIGHT_SHIFT',
  162: 'STABLEHLO_LOGISTIC',
  163: 'STABLEHLO_ADD',
  170: 'STABLEHLO_MULTIPLY',
  175: 'STABLEHLO_REDUCE_WINDOW',
  180: 'STABLEHLO_SCATTER',
};

// TFLite Padding enum
const Padding: Record<number, string> = {
  0: 'SAME',
  1: 'VALID',
};

// TFLite ActivationFunctionType enum
const ActivationFunctionType: Record<number, string> = {
  0: 'NONE',
  1: 'RELU',
  2: 'RELU_N1_TO_1',
  3: 'RELU6',
  4: 'TANH',
  5: 'SIGN_BIT',
};

// TensorType ID → name string
function tensorTypeName(typeId: number): string {
  for (const [name, id] of Object.entries(TensorType)) {
    if (id === typeId) return name;
  }
  return `UNKNOWN_${typeId}`;
}

// ──────────────────────────────────────────────────────
// Minimal FlatBuffers binary reader
// ──────────────────────────────────────────────────────

class FBReader {
  private view: DataView;
  private bytes: Uint8Array;

  constructor(buffer: Uint8Array) {
    this.bytes = buffer;
    this.view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  }

  int8(offset: number): number { return this.view.getInt8(offset); }
  uint8(offset: number): number { return this.view.getUint8(offset); }
  int16(offset: number): number { return this.view.getInt16(offset, true); }
  uint16(offset: number): number { return this.view.getUint16(offset, true); }
  int32(offset: number): number { return this.view.getInt32(offset, true); }
  uint32(offset: number): number { return this.view.getUint32(offset, true); }
  float32(offset: number): number { return this.view.getFloat32(offset, true); }
  float64(offset: number): number { return this.view.getFloat64(offset, true); }

  /** Read a string at the given offset (FlatBuffers string format) */
  string(offset: number): string {
    const len = this.int32(offset);
    const start = offset + 4;
    return new TextDecoder().decode(this.bytes.subarray(start, start + len));
  }

  /** Follow an indirect offset (vtable pattern) */
  indirect(offset: number): number {
    return offset + this.int32(offset);
  }

  /** Get the vtable for a table at the given offset */
  vtable(tableOffset: number): number {
    return tableOffset - this.int32(tableOffset);
  }

  /** Get a field offset from a vtable, returns 0 if not present */
  field(tableOffset: number, fieldIndex: number): number {
    const vtableOffset = this.vtable(tableOffset);
    const vtableSize = this.int16(vtableOffset);
    // Fields start at vtable offset 4, each field is 2 bytes
    const fieldByteOffset = 4 + fieldIndex * 2;
    if (fieldByteOffset >= vtableSize) return 0;
    const relOffset = this.uint16(vtableOffset + fieldByteOffset);
    return relOffset === 0 ? 0 : tableOffset + relOffset;
  }

  /** Read a vector length */
  vectorLen(offset: number): number {
    const vecOffset = this.indirect(offset);
    return this.int32(vecOffset);
  }

  /** Get the start of vector elements */
  vectorStart(offset: number): number {
    return this.indirect(offset) + 4;
  }

  /** Get raw bytes slice */
  slice(start: number, end: number): Uint8Array {
    return this.bytes.subarray(start, end);
  }

  /** Root table offset */
  rootOffset(): number {
    return this.indirect(0);
  }
}

// ──────────────────────────────────────────────────────
// TFLite Model structure readers
// ──────────────────────────────────────────────────────

interface TFLiteTensor {
  name: string;
  shape: number[];
  shapeSignature?: number[];  // from shape_signature field; -1 = dynamic dim
  type: number;        // TensorType enum value
  buffer: number;      // buffer index
  quantization?: {
    scale?: number[];
    zeroPoint?: number[];
    min?: number[];
    max?: number[];
    quantizedDimension?: number;
  };
}

interface TFLiteOperator {
  opcodeIndex: number;
  inputs: number[];    // tensor indices
  outputs: number[];   // tensor indices
  builtinOptionsType: number;
  builtinOptionsOffset: number;
}

function readModel(fb: FBReader): {
  version: number;
  operatorCodes: { builtinCode: number; customCode: string }[];
  subgraphs: {
    name: string;
    tensors: TFLiteTensor[];
    operators: TFLiteOperator[];
    inputs: number[];
    outputs: number[];
  }[];
  buffers: { data: Uint8Array | null }[];
  description: string;
} {
  const root = fb.rootOffset();

  // Model fields: version(0), operator_codes(1), subgraphs(2), description(3), buffers(4)
  const versionField = fb.field(root, 0);
  const version = versionField ? fb.uint32(versionField) : 3;

  const description = readString(fb, root, 3);

  // Operator codes
  const operatorCodes = readOperatorCodes(fb, root);

  // Buffers
  const buffers = readBuffers(fb, root);

  // Subgraphs
  const subgraphs = readSubgraphs(fb, root);

  return { version, operatorCodes, subgraphs, buffers, description };
}

function readString(fb: FBReader, tableOffset: number, fieldIndex: number): string {
  const offset = fb.field(tableOffset, fieldIndex);
  if (!offset) return '';
  return fb.string(fb.indirect(offset));
}

function readOperatorCodes(fb: FBReader, root: number): { builtinCode: number; customCode: string }[] {
  const vecField = fb.field(root, 1);
  if (!vecField) return [];

  const len = fb.vectorLen(vecField);
  const start = fb.vectorStart(vecField);
  const codes: { builtinCode: number; customCode: string }[] = [];

  for (let i = 0; i < len; i++) {
    const entryOffset = fb.indirect(start + i * 4);
    // OperatorCode fields: deprecated_builtin_code(0), custom_code(1), version(2), builtin_code(3)
    const deprecatedCode = fb.field(entryOffset, 0);
    const deprecatedBuiltinCode = deprecatedCode ? fb.int8(deprecatedCode) : 0;

    const builtinCodeField = fb.field(entryOffset, 3);
    // builtin_code is int32, added in schema v3a as field index 3
    const builtinCode = builtinCodeField ? fb.int32(builtinCodeField) : deprecatedBuiltinCode;

    const customCode = readString(fb, entryOffset, 1);

    codes.push({ builtinCode, customCode });
  }

  return codes;
}

function readBuffers(fb: FBReader, root: number): { data: Uint8Array | null }[] {
  const vecField = fb.field(root, 4);
  if (!vecField) return [];

  const len = fb.vectorLen(vecField);
  const start = fb.vectorStart(vecField);
  const buffers: { data: Uint8Array | null }[] = [];

  for (let i = 0; i < len; i++) {
    const entryOffset = fb.indirect(start + i * 4);
    // Buffer fields: data(0), offset(1), size(2)
    const dataField = fb.field(entryOffset, 0);
    if (dataField) {
      const dataLen = fb.vectorLen(dataField);
      const dataStart = fb.vectorStart(dataField);
      buffers.push({ data: fb.slice(dataStart, dataStart + dataLen) });
    } else {
      buffers.push({ data: null });
    }
  }

  return buffers;
}

function readSubgraphs(fb: FBReader, root: number): {
  name: string;
  tensors: TFLiteTensor[];
  operators: TFLiteOperator[];
  inputs: number[];
  outputs: number[];
}[] {
  const vecField = fb.field(root, 2);
  if (!vecField) return [];

  const len = fb.vectorLen(vecField);
  const start = fb.vectorStart(vecField);
  const subgraphs: ReturnType<typeof readSubgraphs> = [];

  for (let i = 0; i < len; i++) {
    const sgOffset = fb.indirect(start + i * 4);
    // SubGraph fields: tensors(0), inputs(1), outputs(2), operators(3), name(4)
    const tensors = readTensors(fb, sgOffset);
    const inputs = readIntVector(fb, sgOffset, 1);
    const outputs = readIntVector(fb, sgOffset, 2);
    const operators = readOperators(fb, sgOffset);
    const name = readString(fb, sgOffset, 4);

    subgraphs.push({ name, tensors, operators, inputs, outputs });
  }

  return subgraphs;
}

function readTensors(fb: FBReader, sgOffset: number): TFLiteTensor[] {
  const vecField = fb.field(sgOffset, 0);
  if (!vecField) return [];

  const len = fb.vectorLen(vecField);
  const start = fb.vectorStart(vecField);
  const tensors: TFLiteTensor[] = [];

  for (let i = 0; i < len; i++) {
    const tOffset = fb.indirect(start + i * 4);
    // Tensor fields: shape(0), type(1), buffer(2), name(3), quantization(4),
    //                is_variable(5), sparsity(6), shape_signature(7)
    const shape = readIntVector(fb, tOffset, 0);
    const typeField = fb.field(tOffset, 1);
    const type = typeField ? fb.int8(typeField) : 0;
    const bufferField = fb.field(tOffset, 2);
    const buffer = bufferField ? fb.uint32(bufferField) : 0;
    const name = readString(fb, tOffset, 3);

    // Read quantization parameters
    const quantField = fb.field(tOffset, 4);
    let quantization: TFLiteTensor['quantization'];
    if (quantField) {
      const qOffset = fb.indirect(quantField);
      quantization = readQuantization(fb, qOffset);
    }

    // Read shape_signature (field 7) — indicates dynamic dimensions with -1
    const sigVec = readIntVector(fb, tOffset, 7);
    const shapeSignature = sigVec.length > 0 ? sigVec : undefined;

    tensors.push({ name, shape, type, buffer, quantization, shapeSignature });
  }

  return tensors;
}

function readQuantization(fb: FBReader, qOffset: number): TFLiteTensor['quantization'] {
  // QuantizationParameters fields: min(0), max(1), scale(2), zero_point(3), quantized_dimension(4)
  const result: TFLiteTensor['quantization'] = {};

  const minField = fb.field(qOffset, 0);
  if (minField) result.min = readFloatVector(fb, minField);

  const maxField = fb.field(qOffset, 1);
  if (maxField) result.max = readFloatVector(fb, maxField);

  const scaleField = fb.field(qOffset, 2);
  if (scaleField) result.scale = readFloatVector(fb, scaleField);

  const zpField = fb.field(qOffset, 3);
  if (zpField) result.zeroPoint = readInt64AsNumberVector(fb, zpField);

  const qdField = fb.field(qOffset, 4);
  // quantized_dimension defaults to 0 in FlatBuffers (the default value is
  // omitted from the vtable), so always store it.
  result.quantizedDimension = qdField ? fb.int32(qdField) : 0;

  return result;
}

function readOperators(fb: FBReader, sgOffset: number): TFLiteOperator[] {
  const vecField = fb.field(sgOffset, 3);
  if (!vecField) return [];

  const len = fb.vectorLen(vecField);
  const start = fb.vectorStart(vecField);
  const operators: TFLiteOperator[] = [];

  for (let i = 0; i < len; i++) {
    const opOffset = fb.indirect(start + i * 4);
    // Operator fields: opcode_index(0), inputs(1), outputs(2), builtin_options_type(3), builtin_options(4)
    const opcodeIndexField = fb.field(opOffset, 0);
    const opcodeIndex = opcodeIndexField ? fb.uint32(opcodeIndexField) : 0;

    const inputs = readIntVector(fb, opOffset, 1);
    const outputs = readIntVector(fb, opOffset, 2);

    const builtinOptionsTypeField = fb.field(opOffset, 3);
    const builtinOptionsType = builtinOptionsTypeField ? fb.uint8(builtinOptionsTypeField) : 0;

    const builtinOptionsField = fb.field(opOffset, 4);
    const builtinOptionsOffset = builtinOptionsField ? fb.indirect(builtinOptionsField) : 0;

    operators.push({ opcodeIndex, inputs, outputs, builtinOptionsType, builtinOptionsOffset });
  }

  return operators;
}

function readIntVector(fb: FBReader, tableOffset: number, fieldIndex: number): number[] {
  const field = fb.field(tableOffset, fieldIndex);
  if (!field) return [];
  const len = fb.vectorLen(field);
  const start = fb.vectorStart(field);
  const result: number[] = [];
  for (let i = 0; i < len; i++) {
    result.push(fb.int32(start + i * 4));
  }
  return result;
}

function readFloatVector(fb: FBReader, field: number): number[] {
  const len = fb.vectorLen(field);
  const start = fb.vectorStart(field);
  const result: number[] = [];
  for (let i = 0; i < len; i++) {
    result.push(fb.float32(start + i * 4));
  }
  return result;
}

function readInt64AsNumberVector(fb: FBReader, field: number): number[] {
  const len = fb.vectorLen(field);
  const start = fb.vectorStart(field);
  const result: number[] = [];
  for (let i = 0; i < len; i++) {
    // Read as two 32-bit values (little-endian) — only use low 32 bits
    result.push(fb.int32(start + i * 8));
  }
  return result;
}

// ──────────────────────────────────────────────────────
// Builtin options parsers — extract attributes from operator tables
// ──────────────────────────────────────────────────────

// BuiltinOptions type enum mapping to parser
type OptionsParser = (fb: FBReader, offset: number) => Record<string, unknown>;

function readConv2DOptions(fb: FBReader, offset: number): Record<string, unknown> {
  // Conv2DOptions: padding(0), stride_w(1), stride_h(2), fused_activation_function(3),
  //   dilation_w_factor(4), dilation_h_factor(5), quantized_bias_type(6)
  const paddingField = fb.field(offset, 0);
  const padding = paddingField ? Padding[fb.int8(paddingField)] ?? 'SAME' : 'SAME';

  const swField = fb.field(offset, 1);
  const stride_w = swField ? fb.int32(swField) : 1;
  const shField = fb.field(offset, 2);
  const stride_h = shField ? fb.int32(shField) : 1;

  const actField = fb.field(offset, 3);
  const fused_activation = actField ? ActivationFunctionType[fb.int8(actField)] ?? 'NONE' : 'NONE';

  const dwField = fb.field(offset, 4);
  const dilation_w_factor = dwField ? fb.int32(dwField) : 1;
  const dhField = fb.field(offset, 5);
  const dilation_h_factor = dhField ? fb.int32(dhField) : 1;

  return { padding, stride_w, stride_h, dilation_w_factor, dilation_h_factor, fused_activation };
}

function readDepthwiseConv2DOptions(fb: FBReader, offset: number): Record<string, unknown> {
  // Same as Conv2D + depth_multiplier(4)
  const paddingField = fb.field(offset, 0);
  const padding = paddingField ? Padding[fb.int8(paddingField)] ?? 'SAME' : 'SAME';
  const swField = fb.field(offset, 1);
  const stride_w = swField ? fb.int32(swField) : 1;
  const shField = fb.field(offset, 2);
  const stride_h = shField ? fb.int32(shField) : 1;

  const dmField = fb.field(offset, 3);
  const depth_multiplier = dmField ? fb.int32(dmField) : 1;

  const actField = fb.field(offset, 4);
  const fused_activation = actField ? ActivationFunctionType[fb.int8(actField)] ?? 'NONE' : 'NONE';

  const dwField = fb.field(offset, 5);
  const dilation_w_factor = dwField ? fb.int32(dwField) : 1;
  const dhField = fb.field(offset, 6);
  const dilation_h_factor = dhField ? fb.int32(dhField) : 1;

  return { padding, stride_w, stride_h, depth_multiplier, dilation_w_factor, dilation_h_factor, fused_activation };
}

function readPool2DOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const paddingField = fb.field(offset, 0);
  // Padding enum: SAME=0, VALID=1. FlatBuffers omits default (0) from vtable,
  // so when field is absent the default is SAME — matching the Conv2D options.
  const padding = paddingField ? Padding[fb.int8(paddingField)] ?? 'SAME' : 'SAME';
  const swField = fb.field(offset, 1);
  const stride_w = swField ? fb.int32(swField) : 1;
  const shField = fb.field(offset, 2);
  const stride_h = shField ? fb.int32(shField) : 1;
  const fwField = fb.field(offset, 3);
  const filter_width = fwField ? fb.int32(fwField) : 1;
  const fhField = fb.field(offset, 4);
  const filter_height = fhField ? fb.int32(fhField) : 1;
  const actField = fb.field(offset, 5);
  const fused_activation = actField ? ActivationFunctionType[fb.int8(actField)] ?? 'NONE' : 'NONE';

  return { padding, stride_w, stride_h, filter_width, filter_height, fused_activation };
}

function readReshapeOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const newShapeField = fb.field(offset, 0);
  if (!newShapeField) return {};
  const len = fb.vectorLen(newShapeField);
  const start = fb.vectorStart(newShapeField);
  const new_shape: number[] = [];
  for (let i = 0; i < len; i++) {
    new_shape.push(fb.int32(start + i * 4));
  }
  return { new_shape };
}

function readSoftmaxOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const betaField = fb.field(offset, 0);
  return { beta: betaField ? fb.float32(betaField) : 1.0 };
}

function readConcatenationOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const axisField = fb.field(offset, 0);
  const axis = axisField ? fb.int32(axisField) : 0;
  const actField = fb.field(offset, 1);
  const fused_activation = actField ? ActivationFunctionType[fb.int8(actField)] ?? 'NONE' : 'NONE';
  return { axis, fused_activation };
}

function readAddOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const actField = fb.field(offset, 0);
  const fused_activation = actField ? ActivationFunctionType[fb.int8(actField)] ?? 'NONE' : 'NONE';
  return { fused_activation };
}

function readFullyConnectedOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const actField = fb.field(offset, 0);
  const fused_activation = actField ? ActivationFunctionType[fb.int8(actField)] ?? 'NONE' : 'NONE';
  const wfField = fb.field(offset, 1);
  const weights_format = wfField ? fb.int8(wfField) : 0;
  const kbField = fb.field(offset, 2);
  const keep_num_dims = kbField ? !!fb.int8(kbField) : false;
  return { fused_activation, weights_format, keep_num_dims };
}

function readResizeBilinearOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const alignField = fb.field(offset, 0);
  const align_corners = alignField ? !!fb.int8(alignField) : false;
  const halfField = fb.field(offset, 1);
  const half_pixel_centers = halfField ? !!fb.int8(halfField) : false;
  return { align_corners, half_pixel_centers };
}

function readPadOptions(_fb: FBReader, _offset: number): Record<string, unknown> {
  // Pad has no options in the options table; padding values come from the input tensor
  return {};
}

function readTransposeOptions(_fb: FBReader, _offset: number): Record<string, unknown> {
  return {};
}

function readSliceOptions(_fb: FBReader, _offset: number): Record<string, unknown> {
  return {};
}

function readStridedSliceOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const bmField = fb.field(offset, 0);
  const begin_mask = bmField ? fb.int32(bmField) : 0;
  const emField = fb.field(offset, 1);
  const end_mask = emField ? fb.int32(emField) : 0;
  const eaField = fb.field(offset, 2);
  const ellipsis_mask = eaField ? fb.int32(eaField) : 0;
  const naField = fb.field(offset, 3);
  const new_axis_mask = naField ? fb.int32(naField) : 0;
  const smField = fb.field(offset, 4);
  const shrink_axis_mask = smField ? fb.int32(smField) : 0;
  return { begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask };
}

function readSplitOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const numField = fb.field(offset, 0);
  return { num_splits: numField ? fb.int32(numField) : 1 };
}

function readTransposeConvOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const paddingField = fb.field(offset, 0);
  const padding = paddingField ? Padding[fb.int8(paddingField)] ?? 'SAME' : 'SAME';
  const swField = fb.field(offset, 1);
  const stride_w = swField ? fb.int32(swField) : 1;
  const shField = fb.field(offset, 2);
  const stride_h = shField ? fb.int32(shField) : 1;
  return { padding, stride_w, stride_h };
}

function readGatherOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const axisField = fb.field(offset, 0);
  return { axis: axisField ? fb.int32(axisField) : 0 };
}

function readLeakyReluOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const alphaField = fb.field(offset, 0);
  return { alpha: alphaField ? fb.float32(alphaField) : 0.01 };
}

function readResizeNearestNeighborOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const alignField = fb.field(offset, 0);
  const align_corners = alignField ? !!fb.int8(alignField) : false;
  const halfField = fb.field(offset, 1);
  const half_pixel_centers = halfField ? !!fb.int8(halfField) : false;
  return { align_corners, half_pixel_centers };
}

function readMeanOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const keepField = fb.field(offset, 0);
  return { keep_dims: keepField ? !!fb.int8(keepField) : false };
}

function readReducerOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const keepField = fb.field(offset, 0);
  return { keep_dims: keepField ? !!fb.int8(keepField) : false };
}

function readSqueezeOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const axisField = fb.field(offset, 0);
  if (!axisField) return {};
  const len = fb.vectorLen(axisField);
  const start = fb.vectorStart(axisField);
  const squeeze_dims: number[] = [];
  for (let i = 0; i < len; i++) {
    squeeze_dims.push(fb.int32(start + i * 4));
  }
  return { squeeze_dims };
}

function readCastOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const inField = fb.field(offset, 0);
  const in_data_type = inField ? fb.int8(inField) : 0;
  const outField = fb.field(offset, 1);
  const out_data_type = outField ? fb.int8(outField) : 0;
  return { in_data_type: tensorTypeName(in_data_type), out_data_type: tensorTypeName(out_data_type) };
}

function readPackOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const valuesField = fb.field(offset, 0);
  const values_count = valuesField ? fb.int32(valuesField) : 0;
  const axisField = fb.field(offset, 1);
  const axis = axisField ? fb.int32(axisField) : 0;
  return { values_count, axis };
}

function readUnpackOptions(fb: FBReader, offset: number): Record<string, unknown> {
  const numField = fb.field(offset, 0);
  const num = numField ? fb.int32(numField) : 0;
  const axisField = fb.field(offset, 1);
  const axis = axisField ? fb.int32(axisField) : 0;
  return { num, axis };
}

// BuiltinOperator code → options parser
//
// DESIGN DECISION: We map by BuiltinOperator enum value, NOT by builtinOptionsType.
//
// In the TFLite FlatBuffers schema, each Operator has two separate numbering systems:
//   1. `opcode_index` → resolves to a BuiltinOperator enum (ADD=0, AVERAGE_POOL_2D=1, CONV_2D=3, ...)
//   2. `builtin_options_type` → a FlatBuffers union index into the BuiltinOptions union
//
// The BuiltinOptions union is declared in schema.fbs as:
//   union BuiltinOptions { Conv2DOptions, DepthwiseConv2DOptions, ... Pool2DOptions, ... }
// The union type index is the 1-based position in this declaration, which does NOT match
// the BuiltinOperator enum values. For example:
//   - Conv2DOptions is union index 1, but CONV_2D is BuiltinOperator 3
//   - Pool2DOptions is union index 5, but AVERAGE_POOL_2D is BuiltinOperator 1
//   - TransposeConvOptions is union index 49, but TRANSPOSE_CONV is BuiltinOperator 67
//
// Mapping by BuiltinOperator code (approach used here) is more robust because:
//   - The enum values are stored in the file itself (via OperatorCode table)
//   - The mapping from op to its options parser is obvious and self-documenting
//   - No need to manually count positions in a 120+ element union declaration
//
// Schema reference:
//   https://github.com/google-ai-edge/LiteRT/blob/main/tflite/converter/schema/schema.fbs
//   (LiteRT is the successor to TensorFlow Lite)
//
const operatorOptionsParsers: Record<number, OptionsParser> = {
  0: readAddOptions,              // ADD
  1: readPool2DOptions,           // AVERAGE_POOL_2D
  2: readConcatenationOptions,    // CONCATENATION
  3: readConv2DOptions,           // CONV_2D
  4: readDepthwiseConv2DOptions,  // DEPTHWISE_CONV_2D
  9: readFullyConnectedOptions,   // FULLY_CONNECTED
  12: readPool2DOptions,          // L2_POOL_2D (same Pool2DOptions)
  17: readPool2DOptions,          // MAX_POOL_2D
  18: readAddOptions,             // MUL (same layout as AddOptions)
  22: readReshapeOptions,         // RESHAPE
  23: readResizeBilinearOptions,  // RESIZE_BILINEAR
  25: readSoftmaxOptions,         // SOFTMAX
  34: readPadOptions,             // PAD
  36: readGatherOptions,          // GATHER
  39: readTransposeOptions,       // TRANSPOSE
  40: readMeanOptions,            // MEAN
  41: readAddOptions,             // SUB (same layout)
  42: readAddOptions,             // DIV (same layout)
  43: readSqueezeOptions,         // SQUEEZE
  45: readStridedSliceOptions,    // STRIDED_SLICE
  49: readSplitOptions,           // SPLIT
  53: readCastOptions,            // CAST
  56: readReducerOptions,         // ARG_MAX
  60: readPadOptions,             // PADV2
  65: readSliceOptions,           // SLICE
  67: readTransposeConvOptions,   // TRANSPOSE_CONV
  74: readReducerOptions,         // SUM
  79: readReducerOptions,         // ARG_MIN
  81: readReducerOptions,         // REDUCE_PROD
  82: readReducerOptions,         // REDUCE_MAX
  83: readPackOptions,            // PACK
  88: readUnpackOptions,          // UNPACK
  89: readReducerOptions,         // REDUCE_MIN
  91: readReducerOptions,         // REDUCE_ANY
  97: readResizeNearestNeighborOptions, // RESIZE_NEAREST_NEIGHBOR
  98: readLeakyReluOptions,       // LEAKY_RELU
};

// ──────────────────────────────────────────────────────
// Main parser: TFLite → GraphIR
// ──────────────────────────────────────────────────────

export async function parseTflite(buffer: Uint8Array): Promise<GraphIR> {
  const fb = new FBReader(buffer);

  // Validate file identifier
  if (buffer.length >= 8) {
    const id = String.fromCharCode(buffer[4], buffer[5], buffer[6], buffer[7]);
    if (id !== 'TFL3') {
      // Not all TFLite files have the identifier, but warn if something else is present
      // and it doesn't look like a flatbuffer
      const rootOffset = fb.int32(0);
      if (rootOffset <= 0 || rootOffset >= buffer.length) {
        throw new Error('Invalid TFLite model: bad root table offset');
      }
    }
  }

  const model = readModel(fb);

  if (model.subgraphs.length === 0) {
    throw new Error('Invalid TFLite model: no subgraphs found');
  }

  // Use the first subgraph (main graph)
  const sg = model.subgraphs[0];

  // Build a set of input/output tensor indices
  const inputIndices = new Set(sg.inputs);
  const outputIndices = new Set(sg.outputs);

  // Determine which tensors are constants (have buffer data and are not graph inputs)
  const constantIndices = new Set<number>();
  for (let i = 0; i < sg.tensors.length; i++) {
    if (inputIndices.has(i) || outputIndices.has(i)) continue;
    const t = sg.tensors[i];
    const buf = model.buffers[t.buffer];
    if (buf?.data && buf.data.length > 0) {
      constantIndices.add(i);
    }
  }

  // Also check graph outputs with buffer data — they should still be outputs
  // Some intermediate tensors may have no buffer and aren't inputs — those are activations

  // Convert tensors
  const tensorNames = sg.tensors.map((t) => t.name || `tensor_${sg.tensors.indexOf(t)}`);

  // Build a mapping from dynamic dim positions (in graph inputs) to symbolic names.
  // TFLite's shape_signature uses -1 for dynamic dims; shape uses a placeholder (often 1).
  // We assign symbolic names so the user can provide overrides via freeDimensionOverrides.
  const dynamicDimNames = new Map<number, string>(); // dim position → name
  let nextDimId = 0;
  for (const idx of sg.inputs) {
    const t = sg.tensors[idx];
    if (!t.shapeSignature) continue;
    for (let d = 0; d < t.shapeSignature.length; d++) {
      if (t.shapeSignature[d] === -1 && !dynamicDimNames.has(d)) {
        dynamicDimNames.set(d, `dim_${nextDimId++}`);
      }
    }
  }

  /** Convert a tensor shape to (number|string)[] using shape_signature to detect dynamic dims */
  function toDynamicShape(t: TFLiteTensor): (number | string)[] {
    if (!t.shapeSignature || dynamicDimNames.size === 0) {
      return t.shape.length > 0 ? t.shape : [];
    }
    return t.shapeSignature.map((s, i) => {
      if (s === -1) {
        const name = dynamicDimNames.get(i);
        return name ?? `dim_${i}`;
      }
      return s;
    });
  }

  // Graph inputs
  const inputs: TensorInfo[] = sg.inputs.map((idx) => {
    const t = sg.tensors[idx];
    return {
      name: tensorNames[idx],
      dataType: tfliteDataTypeFromId(t.type),
      shape: toDynamicShape(t),
    };
  });

  // Graph outputs — use concrete metadata shapes, NOT symbolic dims.
  // Output dynamic dims (e.g. number of detections) are unrelated to input dynamic
  // dims (e.g. height/width), so they must not share the same symbolic names.
  const outputs: TensorInfo[] = sg.outputs.map((idx) => {
    const t = sg.tensors[idx];
    return {
      name: tensorNames[idx],
      dataType: tfliteDataTypeFromId(t.type),
      shape: t.shape.length > 0 ? t.shape : [],
    };
  });

  // Constants
  const constants: ConstantInfo[] = [];
  for (const idx of constantIndices) {
    const t = sg.tensors[idx];
    const buf = model.buffers[t.buffer];
    if (!buf?.data) continue;
    constants.push({
      name: tensorNames[idx],
      dataType: tfliteDataTypeFromId(t.type),
      shape: t.shape,
      rawData: new Uint8Array(buf.data),
      byteLength: buf.data.length,
    });
  }

  // Operators → NodeIR
  const nodes: NodeIR[] = [];
  for (const op of sg.operators) {
    const opcode = model.operatorCodes[op.opcodeIndex];
    const opType = opcode.customCode || BuiltinOperator[opcode.builtinCode] || `UNKNOWN_${opcode.builtinCode}`;

    // Parse builtin options — map by operator code (not builtinOptionsType index)
    let attributes: Record<string, unknown> = {};
    if (op.builtinOptionsType > 0 && op.builtinOptionsOffset > 0) {
      const parser = operatorOptionsParsers[opcode.builtinCode];
      if (parser) {
        attributes = parser(fb, op.builtinOptionsOffset);
      }
    }

    // Add quantization info for quantized ops
    // Keys use local input position (0, 1, 2) so emitters can look up by node.inputs index
    for (let localIdx = 0; localIdx < op.inputs.length; localIdx++) {
      const inputIdx = op.inputs[localIdx];
      if (inputIdx >= 0 && inputIdx < sg.tensors.length) {
        const t = sg.tensors[inputIdx];
        if (t.quantization?.scale?.length) {
          attributes[`input_${localIdx}_scale`] = t.quantization.scale;
          if (t.quantization.zeroPoint?.length) {
            attributes[`input_${localIdx}_zero_point`] = t.quantization.zeroPoint;
          }
          if (t.quantization.quantizedDimension != null) {
            attributes[`input_${localIdx}_quantized_dimension`] = t.quantization.quantizedDimension;
          }
        }
      }
    }

    nodes.push({
      opType,
      inputs: op.inputs
        .filter((idx) => idx >= 0 && idx < sg.tensors.length)
        .map((idx) => tensorNames[idx]),
      outputs: op.outputs
        .filter((idx) => idx >= 0 && idx < sg.tensors.length)
        .map((idx) => tensorNames[idx]),
      attributes,
    });
  }

  // Build shapes map for all tensors (including scalars with shape [])
  // Only graph inputs get symbolic dims; all other tensors use concrete metadata shapes.
  const inputIndexSet = new Set(sg.inputs);
  const shapes = new Map<string, (number | string)[]>();
  const dataTypes = new Map<string, MLOperandDataType>();
  for (let i = 0; i < sg.tensors.length; i++) {
    const t = sg.tensors[i];
    shapes.set(tensorNames[i], inputIndexSet.has(i) ? toDynamicShape(t) : t.shape);
    dataTypes.set(tensorNames[i], tfliteDataTypeFromId(t.type));
  }

  return {
    name: sg.name || model.description || 'tflite_model',
    format: 'tflite',
    inputs,
    outputs,
    constants,
    nodes,
    shapes,
    dataTypes,
  };
}

/** Map TFLite TensorType ID to WebNN data type */
function tfliteDataTypeFromId(typeId: number): MLOperandDataType {
  const name = tensorTypeName(typeId);
  return tfliteDataType(name);
}
