// ONNX protobuf parser → GraphIR
// Parses .onnx files using onnx-proto and produces a format-agnostic GraphIR.
// Supports external data: when tensors reference external files, the caller
// provides a Map<string, Uint8Array> of filename → data.

import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import type { GraphIR, TensorInfo, ConstantInfo, NodeIR, MLOperandDataType } from '../ir/graph.js';
import { onnxDataType } from '../ir/graph.js';

/**
 * Map from external data filename → file contents.
 * Used when ONNX models store weights in separate files
 * (e.g. model.onnx_data, model.onnx_data_1, etc.)
 */
export type ExternalDataMap = Map<string, Uint8Array>;

type Long = { low: number; high: number; toNumber(): number };

function toLong(v: number | Long): number {
  if (typeof v === 'number') return v;
  return v.toNumber();
}

// ONNX attribute type enum
const ATTR_FLOAT = 1;
const ATTR_INT = 2;
const ATTR_STRING = 3;
const ATTR_FLOATS = 6;
const ATTR_INTS = 7;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parseAttribute(attr: any): unknown {
  switch (attr.type) {
    case ATTR_FLOAT: return attr.f ?? 0;
    case ATTR_INT: return toLong(attr.i as number | Long);
    case ATTR_STRING: {
      if (attr.s instanceof Uint8Array) return new TextDecoder().decode(attr.s);
      return attr.s ?? '';
    }
    case ATTR_FLOATS: return attr.floats ?? [];
    case ATTR_INTS: return (attr.ints ?? []).map((v: any) => toLong(v as number | Long));
    default: return undefined;
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function extractShape(vi: any): (number | string)[] {
  const shape = vi.type?.tensorType?.shape;
  if (!shape?.dim) return [];
  return shape.dim.map((d: any) => {
    if (d.dimParam) return d.dimParam;
    if (d.dimValue != null) return toLong(d.dimValue as number | Long);
    return 'dynamic';
  });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function extractDataType(vi: any): MLOperandDataType {
  const elemType = vi.type?.tensorType?.elemType;
  if (elemType == null) return 'float32';
  return onnxDataType(elemType);
}

// Get raw bytes from an ONNX TensorProto
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function extractTensorData(tensor: any, externalData?: ExternalDataMap): Uint8Array {
  // Check for external data (dataLocation === 1 means EXTERNAL)
  if (tensor.dataLocation === 1 && tensor.externalData?.length > 0) {
    return loadExternalTensorData(tensor, externalData);
  }

  // rawData field — most models store weight data here
  if (tensor.rawData && tensor.rawData.length > 0) {
    // protobufjs may return Buffer (Node.js) — ensure Uint8Array
    return new Uint8Array(tensor.rawData);
  }

  const dataType = onnxDataType(tensor.dataType ?? 1);

  // Data stored in typed fields — pack into raw bytes
  if (tensor.floatData && tensor.floatData.length > 0) {
    const f32 = new Float32Array(tensor.floatData);
    return new Uint8Array(f32.buffer);
  }
  if (tensor.int32Data && tensor.int32Data.length > 0) {
    // ONNX uses int32Data for: INT32, INT16, INT8, UINT8, FLOAT16, BOOL
    if (dataType === 'float16') {
      // Each int32 stores one float16 value — extract as uint16
      const u16 = new Uint16Array(tensor.int32Data.length);
      for (let i = 0; i < tensor.int32Data.length; i++) {
        u16[i] = tensor.int32Data[i] & 0xFFFF;
      }
      return new Uint8Array(u16.buffer);
    }
    if (dataType === 'int8') {
      const i8 = new Int8Array(tensor.int32Data.length);
      for (let i = 0; i < tensor.int32Data.length; i++) {
        i8[i] = tensor.int32Data[i];
      }
      return new Uint8Array(i8.buffer);
    }
    if (dataType === 'uint8') {
      const u8 = new Uint8Array(tensor.int32Data.length);
      for (let i = 0; i < tensor.int32Data.length; i++) {
        u8[i] = tensor.int32Data[i];
      }
      return u8;
    }
    const i32 = new Int32Array(tensor.int32Data);
    return new Uint8Array(i32.buffer);
  }
  if (tensor.int64Data && tensor.int64Data.length > 0) {
    const i64 = new BigInt64Array(
      (tensor.int64Data as (number | Long)[]).map((v) => BigInt(toLong(v)))
    );
    return new Uint8Array(i64.buffer);
  }
  if (tensor.doubleData && tensor.doubleData.length > 0) {
    // Downcast double → float32
    if (dataType === 'float32') {
      const f32 = new Float32Array(tensor.doubleData);
      return new Uint8Array(f32.buffer);
    }
    const f64 = new Float64Array(tensor.doubleData);
    return new Uint8Array(f64.buffer);
  }
  if (tensor.uint64Data && tensor.uint64Data.length > 0) {
    const u64 = new BigUint64Array(
      (tensor.uint64Data as (number | Long)[]).map((v) => BigInt(toLong(v)))
    );
    return new Uint8Array(u64.buffer);
  }

  return new Uint8Array(0);
}

/**
 * Load tensor data from external files.
 * External data entries contain key-value pairs:
 *   - "location": relative file path (required)
 *   - "offset": byte offset within the file (optional, default 0)
 *   - "length": number of bytes (optional, default = rest of file)
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function loadExternalTensorData(tensor: any, externalData?: ExternalDataMap): Uint8Array {
  const entries: Record<string, string> = {};
  for (const entry of tensor.externalData) {
    entries[entry.key] = entry.value;
  }

  const location = entries['location'];
  if (!location) {
    throw new Error(`External data tensor "${tensor.name}" missing "location" field`);
  }

  // Normalize path: strip leading "./" 
  const normalizedPath = location.replace(/^\.\//, '');

  if (!externalData) {
    throw new Error(
      `Tensor "${tensor.name}" uses external data file "${normalizedPath}" ` +
      `but no external data was provided. ` +
      `Pass the external data files alongside the model.`
    );
  }

  // Try exact match first, then normalized path
  const fileData = externalData.get(location) ?? externalData.get(normalizedPath);
  if (!fileData) {
    const available = [...externalData.keys()].join(', ');
    throw new Error(
      `External data file "${normalizedPath}" not found for tensor "${tensor.name}". ` +
      `Available files: ${available || '(none)'}`
    );
  }

  const offset = entries['offset'] ? parseInt(entries['offset'], 10) : 0;
  const length = entries['length']
    ? parseInt(entries['length'], 10)
    : fileData.byteLength - offset;

  return new Uint8Array(fileData.buffer, fileData.byteOffset + offset, length);
}

/**
 * Scan an ONNX model buffer for external data file references WITHOUT fully
 * parsing the model. Returns the set of unique file paths referenced.
 * Useful for the CLI/Web UI to know which files to load before calling parseOnnx.
 */
export function getExternalDataRefs(buffer: Uint8Array): string[] {
  const decoded = onnx.ModelProto.decode(buffer);
  const graph = decoded.graph;
  if (!graph) return [];

  const paths = new Set<string>();
  for (const init of graph.initializer ?? []) {
    if (init.dataLocation === 1 && init.externalData?.length) {
      for (const entry of init.externalData) {
        if (entry.key === 'location' && entry.value) {
          paths.add(entry.value.replace(/^\.\//, ''));
        }
      }
    }
  }
  return [...paths];
}

export async function parseOnnx(
  buffer: Uint8Array,
  externalData?: ExternalDataMap,
): Promise<GraphIR> {
  // Detect Git LFS pointer files — raw.githubusercontent.com returns these instead
  // of actual binary content for LFS-tracked files. They start with:
  //   "version https://git-lfs.github.com/spec/v1"
  const LFS_MAGIC = 'version https://git-lfs.github.com/spec/v1';
  const prefix = new TextDecoder().decode(buffer.slice(0, LFS_MAGIC.length));
  if (prefix === LFS_MAGIC) {
    throw new Error(
      'This file is a Git LFS pointer, not the actual model. ' +
      'GitHub raw URLs return LFS pointers instead of file contents for large files. ' +
      'To get the real model: use a HuggingFace URL, download it via "git lfs pull", ' +
      'or find a direct CDN download link.',
    );
  }

  const decoded = onnx.ModelProto.decode(buffer);
  const graph = decoded.graph;

  if (!graph) {
    throw new Error('Invalid ONNX model: no graph found');
  }

  // Collect initializer names (these are constants, not graph inputs)
  const initializerNames = new Set(
    (graph.initializer ?? []).map((init) => init.name ?? '')
  );

  // Parse inputs (exclude initializers that also appear in graph.input)
  const inputs: TensorInfo[] = (graph.input ?? [])
    .filter((inp) => !initializerNames.has(inp.name ?? ''))
    .map((inp) => ({
      name: inp.name ?? '',
      dataType: extractDataType(inp),
      shape: extractShape(inp),
    }));

  // Parse outputs
  const outputs: TensorInfo[] = (graph.output ?? []).map((out) => ({
    name: out.name ?? '',
    dataType: extractDataType(out),
    shape: extractShape(out),
  }));

  // Parse constants (initializers)
  const constants: ConstantInfo[] = (graph.initializer ?? []).map((init) => {
    const rawData = extractTensorData(init, externalData);
    return {
      name: init.name ?? '',
      dataType: onnxDataType(init.dataType ?? 1),
      shape: (init.dims ?? []).map((d) => toLong(d as number | Long)),
      rawData,
      byteLength: rawData.byteLength,
    };
  });

  // Extract ONNX Constant ops: these embed tensor data as an attribute and
  // should be treated as constants, not graph operations.
  // ORT handles these during graph preprocessing (PreprocessInitializers), not
  // as runtime ops — there is no WebNN Constant op builder.
  const constantNodeNames = new Set<string>();
  for (const node of graph.node ?? []) {
    if (node.opType !== 'Constant') continue;
    const outputName = node.output?.[0] ?? '';
    if (!outputName) continue;

    // The "value" attribute is type TENSOR (attr.type === 4) stored in attr.t
    const valueAttr = (node.attribute ?? []).find((a: any) => a.name === 'value');
    if (valueAttr?.t) {
      const tensor = valueAttr.t;
      const rawData = extractTensorData(tensor, externalData);
      constants.push({
        name: outputName,
        dataType: onnxDataType(tensor.dataType ?? 1),
        shape: (tensor.dims ?? []).map((d: any) => toLong(d as number | Long)),
        rawData,
        byteLength: rawData.byteLength,
      });
      constantNodeNames.add(outputName);
    } else {
      // Handle value_int, value_float, value_ints, value_floats, value_string scalar attrs
      const intAttr = (node.attribute ?? []).find((a: any) => a.name === 'value_int');
      const floatAttr = (node.attribute ?? []).find((a: any) => a.name === 'value_float');
      if (intAttr && intAttr.i != null) {
        const val = BigInt(toLong(intAttr.i as number | Long));
        const rawData = new Uint8Array(new BigInt64Array([val]).buffer);
        constants.push({ name: outputName, dataType: 'int64', shape: [], rawData, byteLength: 8 });
        constantNodeNames.add(outputName);
      } else if (floatAttr && floatAttr.f != null) {
        const rawData = new Uint8Array(new Float32Array([floatAttr.f]).buffer);
        constants.push({ name: outputName, dataType: 'float32', shape: [], rawData, byteLength: 4 });
        constantNodeNames.add(outputName);
      }
    }
  }

  // Parse nodes (exclude Constant ops — they've been extracted above)
  const nodes: NodeIR[] = (graph.node ?? [])
    .filter((node) => node.opType !== 'Constant')
    .map((node) => {
    const attributes: Record<string, unknown> = {};
    for (const attr of node.attribute ?? []) {
      const value = parseAttribute(attr);
      if (value !== undefined) {
        attributes[attr.name ?? ''] = value;
      }
    }
    return {
      opType: node.opType ?? '',
      inputs: [...(node.input ?? [])],
      outputs: [...(node.output ?? [])],
      attributes,
    };
  });

  // Build shape map from graph inputs, outputs, constants, and value_info
  const shapes = new Map<string, (number | string)[]>();
  const dataTypes = new Map<string, MLOperandDataType>();
  for (const inp of inputs) { shapes.set(inp.name, inp.shape); dataTypes.set(inp.name, inp.dataType); }
  for (const out of outputs) { shapes.set(out.name, out.shape); dataTypes.set(out.name, out.dataType); }
  for (const c of constants) { shapes.set(c.name, c.shape); dataTypes.set(c.name, c.dataType); }
  for (const vi of graph.valueInfo ?? []) {
    const name = vi.name ?? '';
    if (name && !shapes.has(name)) {
      const viShape = extractShape(vi);
      // Skip empty shapes from value_info — they indicate unknown rank, not scalar.
      // True scalars come from constants/initializers which are already in the shapes map.
      if (viShape.length > 0) {
        shapes.set(name, viShape);
      }
      dataTypes.set(name, extractDataType(vi));
    }
  }

  // Propagate shapes through shape-preserving ops when value_info is missing.
  // QDQ models often lack value_info for intermediate QuantizeLinear/DequantizeLinear outputs.
  propagateShapes(nodes, shapes, dataTypes, constants);

  return {
    name: graph.name || 'model',
    format: 'onnx',
    inputs,
    outputs,
    constants,
    nodes,
    shapes,
    dataTypes,
  };
}

/**
 * Propagate shapes through ops for intermediate tensors that have missing or
 * dynamic-dim shapes in value_info. Runs a single forward pass through nodes,
 * overwriting dynamic string dims with concrete numbers when computable.
 */
const SHAPE_PRESERVING_OPS = new Set([
  'QuantizeLinear', 'DequantizeLinear',
  'BatchNormalization', 'LayerNormalization', 'GroupNormalization', 'InstanceNormalization',
  'Relu', 'Sigmoid', 'Tanh', 'Elu', 'Gelu', 'HardSigmoid', 'HardSwish',
  'LeakyRelu', 'Softplus', 'Softsign',
  'Abs', 'Ceil', 'Cos', 'Erf', 'Exp', 'Floor', 'Identity', 'Log',
  'Neg', 'Reciprocal', 'Round', 'Sign', 'Sin', 'Sqrt', 'Tan', 'Not',
  'Cast', 'Dropout', 'Clip', 'Softmax', 'LogSoftmax',
]);

// Element-wise ops where output shape = broadcast(input shapes)
const ELEMENTWISE_OPS = new Set([
  'Add', 'Sub', 'Mul', 'Div', 'Pow', 'PRelu',
  'Equal', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual',
  'Min', 'Max', 'Mean', 'Where',
]);

function isFullyStatic(shape: (number | string)[]): boolean {
  return shape.every(d => typeof d === 'number');
}

/**
 * Build a map from tensor name → int64 values for all constant tensors of type int64.
 * Covers initializers and ONNX Constant op outputs.
 */
function buildConstInt64Map(constants: ConstantInfo[]): Map<string, bigint[]> {
  const map = new Map<string, bigint[]>();
  for (const c of constants) {
    if (c.dataType !== 'int64') continue;
    const view = new DataView(c.rawData.buffer, c.rawData.byteOffset, c.rawData.byteLength);
    const count = c.rawData.byteLength / 8;
    const vals: bigint[] = [];
    for (let i = 0; i < count; i++) {
      vals.push(view.getBigInt64(i * 8, true)); // little-endian
    }
    map.set(c.name, vals);
  }
  return map;
}

/**
 * Attempt to statically evaluate an int64 tensor to a concrete array of bigint values.
 * Recursively traces Shape, Slice, Concat, Gather, Unsqueeze/Reshape, Mul ops.
 * Returns null if any input is unknown or the op is not handled.
 */
function evalIntTensor(
  tensorName: string,
  producerMap: Map<string, NodeIR>,
  constInt64: Map<string, bigint[]>,
  shapes: Map<string, (number | string)[]>,
  cache: Map<string, bigint[] | null>,
  depth = 0,
): bigint[] | null {
  if (depth > 30) return null;
  if (cache.has(tensorName)) return cache.get(tensorName) ?? null;

  const direct = constInt64.get(tensorName);
  if (direct) { cache.set(tensorName, direct); return direct; }

  const producer = producerMap.get(tensorName);
  if (!producer) { cache.set(tensorName, null); return null; }

  const rec = (name: string): bigint[] | null =>
    evalIntTensor(name, producerMap, constInt64, shapes, cache, depth + 1);

  let result: bigint[] | null = null;

  // Constant op (opset 9+): value stored directly in node attribute
  if (producer.opType === 'Constant') {
    const t = producer.attributes.value as { dataType?: string; rawData?: Uint8Array; int64Data?: bigint[]; int32Data?: number[]; floatData?: number[] } | undefined;
    if (t?.int64Data) {
      result = t.int64Data;
    } else if (t?.int32Data) {
      result = t.int32Data.map(BigInt);
    } else if (t?.rawData && (t?.dataType === 'int64' || t?.dataType === 'int32')) {
      const view = new DataView(t.rawData.buffer, t.rawData.byteOffset, t.rawData.byteLength);
      const isI64 = t.dataType === 'int64';
      const stride = isI64 ? 8 : 4;
      const count = t.rawData.byteLength / stride;
      result = Array.from({ length: count }, (_, i) =>
        isI64 ? view.getBigInt64(i * stride, true) : BigInt(view.getInt32(i * stride, true)),
      );
    }
    cache.set(tensorName, result);
    return result;
  }

  if (producer.opType === 'Shape') {
    const inputShape = shapes.get(producer.inputs[0]);
    if (inputShape && inputShape.length > 0 && inputShape.every(d => typeof d === 'number' && (d as number) >= 0)) {
      result = (inputShape as number[]).map(BigInt);
    }
  } else if (producer.opType === 'Slice') {
    const data = rec(producer.inputs[0]);
    if (!data) { cache.set(tensorName, null); return null; }
    const starts = rec(producer.inputs[1]);
    const ends = rec(producer.inputs[2]);
    if (!starts || !ends) { cache.set(tensorName, null); return null; }
    const axesVals = producer.inputs[3] ? rec(producer.inputs[3]) : null;
    const stepsVals = producer.inputs[4] ? rec(producer.inputs[4]) : null;
    // Only handle single-axis slice on a 1D data array
    const axis = axesVals ? Number(axesVals[0]) : 0;
    const normalAxis = axis < 0 ? axis + data.length : axis;
    if (normalAxis !== 0) { cache.set(tensorName, null); return null; }
    const start = Math.max(0, Number(starts[0]) < 0 ? data.length + Number(starts[0]) : Math.min(Number(starts[0]), data.length));
    const end = Math.min(data.length, Number(ends[0]) < 0 ? data.length + Number(ends[0]) : Math.min(Number(ends[0]), data.length));
    const step = stepsVals ? Number(stepsVals[0]) : 1;
    result = [];
    for (let i = start; i < end; i += step) result.push(data[i]);
  } else if (producer.opType === 'Concat') {
    const axisAttr = (producer.attributes.axis as number) ?? 0;
    if (axisAttr !== 0) { cache.set(tensorName, null); return null; }
    const parts: bigint[] = [];
    for (const inp of producer.inputs) {
      if (!inp) { cache.set(tensorName, null); return null; }
      const vals = rec(inp);
      if (!vals) { cache.set(tensorName, null); return null; }
      parts.push(...vals);
    }
    result = parts;
  } else if (producer.opType === 'Gather') {
    const data = rec(producer.inputs[0]);
    if (!data) { cache.set(tensorName, null); return null; }
    const idxVals = rec(producer.inputs[1]);
    if (!idxVals) { cache.set(tensorName, null); return null; }
    const idx = Number(idxVals[0]);
    const normalIdx = idx < 0 ? idx + data.length : idx;
    if (normalIdx < 0 || normalIdx >= data.length) { cache.set(tensorName, null); return null; }
    result = [data[normalIdx]];
  } else if (producer.opType === 'Unsqueeze' || producer.opType === 'Reshape' || producer.opType === 'Squeeze' || producer.opType === 'Flatten') {
    result = rec(producer.inputs[0]);
  } else if (producer.opType === 'Mul') {
    const a = rec(producer.inputs[0]);
    const b = rec(producer.inputs[1]);
    if (!a || !b) { cache.set(tensorName, null); return null; }
    if (a.length === 1 && b.length >= 1) result = b.map(v => a[0] * v);
    else if (b.length === 1 && a.length >= 1) result = a.map(v => v * b[0]);
    else if (a.length === b.length) result = a.map((v, i) => v * b[i]);
  }

  cache.set(tensorName, result);
  return result;
}

/**
 * Resolve ONNX reshape special values in-place:
 *  0  → copy from input dim at same axis (when allowZero=0)
 * -1  → infer from total element count
 * Returns null if resolution fails.
 */
function resolveReshapeShapeLocal(
  targetShape: number[],
  inputShape: (number | string)[],
  allowZero: number,
): number[] | null {
  const resolved = [...targetShape];
  const inputDims = inputShape.length > 0 && inputShape.every(d => typeof d === 'number') ? inputShape as number[] : null;

  if (!allowZero && inputDims) {
    for (let i = 0; i < resolved.length; i++) {
      if (resolved[i] === 0 && i < inputDims.length) resolved[i] = inputDims[i];
    }
  }
  const inferIdx = resolved.indexOf(-1);
  if (inferIdx !== -1) {
    if (!inputDims) return null;
    const totalInput = inputDims.reduce((a, b) => a * b, 1);
    const knownProduct = resolved.reduce((a, b, i) => (i === inferIdx ? a : a * b), 1);
    if (knownProduct <= 0) return null;
    resolved[inferIdx] = totalInput / knownProduct;
  }
  if (resolved.some(d => d < 0 || !Number.isInteger(d))) return null;
  return resolved;
}

/**
 * Compute the broadcast output shape for element-wise ops.
 * Prefers dynamic (string) dims over 1, and prefers larger concrete dims over 1.
 * This avoids the SE-block pattern (Mul of [B,C,H,W] and [1,C,1,1]) producing wrong [1,C,1,1].
 */
function computeBroadcastShape(
  inputShapes: ((number | string)[])[],
): (number | string)[] | null {
  const valid = inputShapes.filter(s => s && s.length > 0);
  if (valid.length === 0) return null;

  const maxRank = Math.max(...valid.map(s => s.length));
  // Pad from the left with 1 to align ranks
  const padded = valid.map(s => {
    const pad = maxRank - s.length;
    return [...Array(pad).fill(1), ...s] as (number | string)[];
  });

  const result: (number | string)[] = [];
  for (let i = 0; i < maxRank; i++) {
    let outDim: number | string = 1;
    for (const s of padded) {
      const d = s[i];
      if (typeof d === 'string') {
        // String (dynamic) dim dominates 1, but if outDim is already a concrete
        // number > 1, keep the concrete number (both should resolve to same value)
        if (outDim === 1) outDim = d;
        // else keep the existing outDim (string or larger concrete number)
      } else if (typeof outDim === 'number') {
        // Both concrete: take the larger (handles 1-broadcasting)
        if ((d as number) > outDim) outDim = d;
      }
      // if outDim is already a string, don't override with a number
    }
    result.push(outDim);
  }
  return result;
}

/**
 * Returns true if `candidate` is a better shape than `existing` for the same tensor.
 * Better means: same rank AND at least one dim is more specific (concrete number or named
 * free-dim symbol like 'batch_size' rather than an anonymous 'unk__*' symbol).
 * This allows propagation to replace value_info's anonymous unk symbols with
 * named free dims propagated forward from graph inputs.
 */
function isBetterShape(
  existing: (number | string)[],
  candidate: (number | string)[],
): boolean {
  if (candidate.length !== existing.length) return false;
  let gain = false;
  for (let i = 0; i < existing.length; i++) {
    const e = existing[i];
    const c = candidate[i];
    if (e === c) continue;
    // Existing is anonymous symbol (unk__*), candidate is concrete or named — gain
    if (typeof e === 'string' && e.startsWith('unk__') &&
        (typeof c === 'number' || (typeof c === 'string' && !c.startsWith('unk__')))) {
      gain = true;
      continue;
    }
    // Existing is concrete but candidate is not — regression, reject
    if (typeof e === 'number' && typeof c !== 'number') return false;
    // Existing is named symbol, candidate is anonymous — regression, reject
    if (typeof e === 'string' && !e.startsWith('unk__') && typeof c === 'string' && c.startsWith('unk__')) return false;
  }
  return gain;
}

function propagateShapes(
  nodes: NodeIR[],
  shapes: Map<string, (number | string)[]>,
  _dataTypes: Map<string, MLOperandDataType>,
  constants: ConstantInfo[] = [],
  forceUpdate = false,
): void {
  // Build lookup tables for constant-folding dynamic shape inputs (Reshape)
  const constInt64 = buildConstInt64Map(constants);
  const producerMap = new Map<string, NodeIR>();
  for (const node of nodes) {
    for (const out of node.outputs) {
      if (out) producerMap.set(out, node);
    }
  }
  const evalCache = new Map<string, bigint[] | null>();
  for (const node of nodes) {
    // Shape-preserving ops: output shape = input shape
    if (SHAPE_PRESERVING_OPS.has(node.opType)) {
      const inputShape = shapes.get(node.inputs[0]);
      if (!inputShape || inputShape.length === 0) continue;
      for (const out of node.outputs) {
        if (!out) continue;
        const existing = shapes.get(out);
        if (!existing || existing.length === 0) {
          shapes.set(out, [...inputShape]);
        } else if (isBetterShape(existing, inputShape) || (forceUpdate || !isFullyStatic(existing)) && isFullyStatic(inputShape)) {
          shapes.set(out, [...inputShape]);
        }
      }
      continue;
    }

    // DynamicQuantizeLinear: output[0] has same shape as input, outputs[1,2] are scalar
    if (node.opType === 'DynamicQuantizeLinear') {
      const inputShape = shapes.get(node.inputs[0]);
      if (inputShape && inputShape.length > 0) {
        const out0 = node.outputs[0];
        if (out0) {
          const existing = shapes.get(out0);
          if (!existing || existing.length === 0) {
            shapes.set(out0, [...inputShape]);
          } else if ((forceUpdate || !isFullyStatic(existing)) && isFullyStatic(inputShape)) {
            shapes.set(out0, [...inputShape]);
          }
        }
      }
      continue;
    }

    // Element-wise binary ops: compute broadcast shape from all inputs
    if (ELEMENTWISE_OPS.has(node.opType)) {
      const inputShapes = node.inputs
        .filter(inp => inp)
        .map(inp => shapes.get(inp!) ?? null)
        .filter(s => s !== null) as (number | string)[][];
      const broadcastShape = computeBroadcastShape(inputShapes);
      if (broadcastShape && broadcastShape.length > 0) {
        for (const out of node.outputs) {
          if (!out) continue;
          const existing = shapes.get(out);
          // Only set if missing, or if the broadcast shape provides more info
          // (higher rank or more concrete dims without losing spatial information)
          if (!existing || existing.length === 0) {
            shapes.set(out, broadcastShape);
          } else if (isBetterShape(existing, broadcastShape)) {
            shapes.set(out, broadcastShape);
          } else if (broadcastShape.length > existing.length) {
            shapes.set(out, broadcastShape);
          } else if ((forceUpdate || !isFullyStatic(existing)) && isFullyStatic(broadcastShape) &&
                     broadcastShape.every((d, i) => {
                       const e = existing[i];
                       return typeof e !== 'number' || e <= 1 || d === e;
                     })) {
            // Only replace with a static shape if it doesn't disagree with existing
            // concrete dims (avoids replacing [B,C,8,8] with [1,C,1,1])
            shapes.set(out, broadcastShape);
          }
        }
      }
      continue;
    }

    // Conv/ConvTranspose: infer output spatial dims from input + attributes
    if (node.opType === 'Conv' || node.opType === 'ConvInteger') {
      const inputShape = shapes.get(node.inputs[0]);
      const weightShape = shapes.get(node.inputs[1]);
      if (!inputShape || !weightShape || inputShape.length < 3) continue;

      const rank = inputShape.length;
      const spatialDims = rank - 2; // N, C, [spatial...]
      const strides = (node.attributes.strides as number[]) ?? Array(spatialDims).fill(1);
      const dilations = (node.attributes.dilations as number[]) ?? Array(spatialDims).fill(1);
      const pads = (node.attributes.pads as number[]) ?? Array(spatialDims * 2).fill(0);
      const outChannels = weightShape[0]; // OIHW weight layout

      const outShape: (number | string)[] = [inputShape[0], outChannels];
      for (let i = 0; i < spatialDims; i++) {
        const inDim = inputShape[i + 2];
        const kernDim = weightShape[i + 2];
        if (typeof inDim === 'number' && typeof kernDim === 'number') {
          const effectiveKernel = (kernDim - 1) * dilations[i] + 1;
          const padded = inDim + pads[i] + pads[i + spatialDims];
          outShape.push(Math.floor((padded - effectiveKernel) / strides[i]) + 1);
        } else {
          outShape.push(typeof inDim === 'string' ? inDim : `d${i}`);
        }
      }

      for (const out of node.outputs) {
        if (!out) continue;
        const existing = shapes.get(out);
        if (!existing || existing.length === 0) {
          shapes.set(out, outShape);
        } else if ((forceUpdate || !isFullyStatic(existing)) && isFullyStatic(outShape)) {
          shapes.set(out, outShape);
        }
      }
      continue;
    }

    // DepthToSpace: output shape from input shape + blocksize
    if (node.opType === 'DepthToSpace') {
      const inputShape = shapes.get(node.inputs[0]);
      if (!inputShape || inputShape.length !== 4) continue;
      const blocksize = (node.attributes.blocksize as number) ?? 1;
      const [b, c, h, w] = inputShape;
      const outShape: (number | string)[] = [
        b,
        typeof c === 'number' ? c / (blocksize * blocksize) : c,
        typeof h === 'number' ? h * blocksize : h,
        typeof w === 'number' ? w * blocksize : w,
      ];
      for (const out of node.outputs) {
        if (!out) continue;
        const existing = shapes.get(out);
        if (!existing || existing.length === 0) {
          shapes.set(out, outShape);
        } else if ((forceUpdate || !isFullyStatic(existing)) && isFullyStatic(outShape)) {
          shapes.set(out, outShape);
        }
      }
      continue;
    }

    // SpaceToDepth: inverse of DepthToSpace
    if (node.opType === 'SpaceToDepth') {
      const inputShape = shapes.get(node.inputs[0]);
      if (!inputShape || inputShape.length !== 4) continue;
      const blocksize = (node.attributes.blocksize as number) ?? 1;
      const [b, c, h, w] = inputShape;
      const outShape: (number | string)[] = [
        b,
        typeof c === 'number' ? c * blocksize * blocksize : c,
        typeof h === 'number' ? Math.floor(h / blocksize) : h,
        typeof w === 'number' ? Math.floor(w / blocksize) : w,
      ];
      for (const out of node.outputs) {
        if (!out) continue;
        const existing = shapes.get(out);
        if (!existing || existing.length === 0) {
          shapes.set(out, outShape);
        } else if ((forceUpdate || !isFullyStatic(existing)) && isFullyStatic(outShape)) {
          shapes.set(out, outShape);
        }
      }
      continue;
    }

    // Flatten: output shape = [product(dims[:axis]), product(dims[axis:])]
    if (node.opType === 'Flatten') {
      const inputShape = shapes.get(node.inputs[0]);
      if (!inputShape || inputShape.length === 0) continue;
      const axis = (node.attributes.axis as number) ?? 1;
      const normalAxis = axis < 0 ? axis + inputShape.length : axis;
      // Can only compute if all dims are numeric
      if (inputShape.every((d): d is number => typeof d === 'number')) {
        const dim0 = (inputShape as number[]).slice(0, normalAxis).reduce((a, b) => a * b, 1);
        const dim1 = (inputShape as number[]).slice(normalAxis).reduce((a, b) => a * b, 1);
        const outShape = [dim0, dim1];
        for (const out of node.outputs) {
          if (!out) continue;
          const existing = shapes.get(out);
          if (!existing || existing.length === 0) {
            shapes.set(out, outShape);
          } else if ((forceUpdate || !isFullyStatic(existing)) && isFullyStatic(outShape)) {
            shapes.set(out, outShape);
          }
        }
      }
      continue;
    }

    // Gather: output shape = indices_shape + data_shape[1:]
    // This covers embedding lookups: Gather(embed_table, input_ids) → [*input_ids_shape, embed_dim]
    if (node.opType === 'Gather') {
      const dataShape = shapes.get(node.inputs[0]);
      const idxShape = shapes.get(node.inputs[1]);
      if (!dataShape || dataShape.length === 0 || !idxShape) continue;
      const axis = (node.attributes.axis as number) ?? 0;
      const normalAxis = axis < 0 ? axis + dataShape.length : axis;
      // Output shape: replace axis dim in dataShape with idxShape dims
      const outShape: (number | string)[] = [
        ...dataShape.slice(0, normalAxis),
        ...idxShape,
        ...dataShape.slice(normalAxis + 1),
      ];
      for (const out of node.outputs) {
        if (!out) continue;
        const existing = shapes.get(out);
        if (!existing || existing.length === 0) {
          shapes.set(out, outShape);
        } else if (isBetterShape(existing, outShape)) {
          shapes.set(out, outShape);
        } else if ((forceUpdate || !isFullyStatic(existing)) && isFullyStatic(outShape)) {
          shapes.set(out, outShape);
        }
      }
      continue;
    }

    // MatMul: output shape = [*batch_dims, M, N]
    // Propagates batch dims from first input (e.g. [batch, seq, M] × [M, N] → [batch, seq, N])
    if (node.opType === 'MatMul') {
      const aShape = shapes.get(node.inputs[0]);
      const bShape = shapes.get(node.inputs[1]);
      if (!aShape || aShape.length < 2 || !bShape || bShape.length < 1) continue;
      const N = bShape[bShape.length - 1];
      // Batch dims from A (all but last 2 dims when rank>2, or nothing when rank=2)
      const batchDims = aShape.length > 2 ? aShape.slice(0, aShape.length - 2) : (aShape.length === 2 ? [] : aShape.slice(0, -1));
      const M = aShape[aShape.length - 2];
      const outShape: (number | string)[] = [...batchDims, M, N];
      for (const out of node.outputs) {
        if (!out) continue;
        const existing = shapes.get(out);
        if (!existing || existing.length === 0) {
          shapes.set(out, outShape);
        } else if (isBetterShape(existing, outShape)) {
          shapes.set(out, outShape);
        } else if ((forceUpdate || !isFullyStatic(existing)) && isFullyStatic(outShape)) {
          shapes.set(out, outShape);
        }
      }
      continue;
    }

    // Transpose: output shape = input shape permuted by perm attribute
    if (node.opType === 'Transpose') {
      const inputShape = shapes.get(node.inputs[0]);
      if (!inputShape || inputShape.length === 0) continue;
      const perm = node.attributes.perm as number[] | undefined;
      let outShape: (number | string)[];
      if (perm && perm.length === inputShape.length) {
        outShape = perm.map(i => inputShape[i]);
      } else {
        // Default: reverse dimensions
        outShape = [...inputShape].reverse();
      }
      for (const out of node.outputs) {
        if (!out) continue;
        const existing = shapes.get(out);
        if (!existing || existing.length === 0) {
          shapes.set(out, outShape);
        } else if ((forceUpdate || !isFullyStatic(existing)) && isFullyStatic(outShape)) {
          shapes.set(out, outShape);
        }
      }
      continue;
    }

    // Reshape: try to resolve output shape using constant-folded shape input
    if (node.opType === 'Reshape') {
      const shapeInput = node.inputs[1];
      if (!shapeInput) continue;
      const inputShape = shapes.get(node.inputs[0]) ?? null;
      // Evaluate the shape tensor using constant folding (handles Shape/Slice/Concat chains)
      const shapeVals = evalIntTensor(shapeInput, producerMap, constInt64, shapes, evalCache);
      if (!shapeVals) continue;
      const allowZero = (node.attributes.allowzero as number) ?? 0;
      // resolveReshapeShapeLocal only needs inputShape when target has 0 or -1 dims;
      // if neither is present, pass null (it's still resolved correctly).
      const resolved = resolveReshapeShapeLocal(
        shapeVals.map(Number),
        inputShape ?? [],
        allowZero,
      );
      if (!resolved) continue;
      // Always trust evalIntTensor result: it's constant-folded from the actual
      // shape-computation graph and supersedes any earlier propagation estimate.
      for (const out of node.outputs) {
        if (out) shapes.set(out, resolved);
      }
      continue;
    }

    // Split: output[i] shape = input shape with split[i] at the split axis
    if (node.opType === 'Split') {
      const inputShape = shapes.get(node.inputs[0]);
      if (!inputShape || inputShape.length === 0) continue;
      const axis = (node.attributes.axis as number) ?? 0;
      const normalAxis = axis < 0 ? axis + inputShape.length : axis;
      if (normalAxis < 0 || normalAxis >= inputShape.length) continue;
      const splitAttr = node.attributes.split as number[] | undefined;
      const numOutputs = node.outputs.filter(Boolean).length;
      let sizes: number[];
      if (splitAttr && splitAttr.length > 0) {
        sizes = splitAttr;
      } else {
        // Equal split
        const dim = inputShape[normalAxis];
        if (typeof dim !== 'number' || dim <= 0) continue;
        const sz = Math.floor(dim / numOutputs);
        sizes = Array(numOutputs).fill(sz);
      }
      for (let i = 0; i < numOutputs; i++) {
        const out = node.outputs[i];
        if (!out) continue;
        const outShape = [...inputShape] as (number | string)[];
        outShape[normalAxis] = sizes[i] ?? inputShape[normalAxis];
        const existing = shapes.get(out);
        if (!existing || existing.length === 0) {
          shapes.set(out, outShape);
        } else if ((forceUpdate || !isFullyStatic(existing)) && isFullyStatic(outShape)) {
          shapes.set(out, outShape);
        }
      }
      continue;
    }

    // Squeeze: output shape = input shape with specified axes removed (or all size-1 dims)
    if (node.opType === 'Squeeze') {
      const inputShape = shapes.get(node.inputs[0]);
      if (!inputShape || inputShape.length === 0) continue;
      // Axes from attribute (opset < 13) or second constant input (opset 13+)
      let axes: number[] | undefined = node.attributes.axes as number[] | undefined;
      if (!axes && node.inputs.length > 1 && node.inputs[1]) {
        const axesConst = constInt64.get(node.inputs[1]);
        if (axesConst) axes = axesConst.map(Number);
      }
      let outShape: (number | string)[];
      if (axes && axes.length > 0) {
        const rank = inputShape.length;
        const resolved = new Set(axes.map((a) => (a < 0 ? a + rank : a)));
        outShape = inputShape.filter((_, i) => !resolved.has(i));
      } else {
        outShape = inputShape.filter((d) => d !== 1);
      }
      for (const out of node.outputs) {
        if (!out) continue;
        const existing = shapes.get(out);
        if (!existing || existing.length === 0) {
          shapes.set(out, outShape);
        } else if ((forceUpdate || !isFullyStatic(existing)) && isFullyStatic(outShape)) {
          shapes.set(out, outShape);
        }
      }
      continue;
    }

    // Unsqueeze: output shape = input shape with 1s inserted at specified axes
    if (node.opType === 'Unsqueeze') {
      const inputShape = shapes.get(node.inputs[0]);
      if (!inputShape || inputShape.length === 0) continue;
      let axes: number[] | undefined = node.attributes.axes as number[] | undefined;
      if (!axes && node.inputs.length > 1 && node.inputs[1]) {
        const axesConst = constInt64.get(node.inputs[1]);
        if (axesConst) axes = axesConst.map(Number);
      }
      if (!axes || axes.length === 0) continue;
      const expandedRank = inputShape.length + axes.length;
      const resolved = axes.map((a) => (a < 0 ? a + expandedRank : a)).sort((a, b) => a - b);
      const outShape: (number | string)[] = [];
      let srcIdx = 0;
      for (let i = 0; i < expandedRank; i++) {
        if (resolved.includes(i)) {
          outShape.push(1);
        } else {
          outShape.push(inputShape[srcIdx++]);
        }
      }
      for (const out of node.outputs) {
        if (!out) continue;
        const existing = shapes.get(out);
        if (!existing || existing.length === 0) {
          shapes.set(out, outShape);
        }
      }
      continue;
    }
  }
}

/**
 * Re-run shape propagation on a graph after free-dimension overrides have been applied.
 * This resolves Reshape outputs whose target shapes depend on dynamic dims (e.g. batch_size)
 * that are only concretized after applyFreeDimensionOverrides.
 */
export function repropagateReshapeShapes(graph: GraphIR): void {
  if (!graph.shapes || !graph.dataTypes) return;
  propagateShapes(graph.nodes, graph.shapes, graph.dataTypes, graph.constants, /*forceUpdate=*/true);
}

