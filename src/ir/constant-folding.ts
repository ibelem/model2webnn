// Constant folding pipeline for GraphIR
// Evaluates nodes whose inputs are all constants at build time, replacing them
// with pre-computed ConstantInfo entries and removing them from the graph.
// This resolves shape-computation chains (Shape → Gather → Concat → Reshape)
// and ops without WebNN equivalents (Range, ConstantOfShape) in transformer models.

import type { GraphIR, ConstantInfo, NodeIR, MLOperandDataType } from './graph.js';
import { bytesPerElement } from './graph.js';

// ---------------------------------------------------------------------------
// Evaluator interface — each foldable op implements this
// ---------------------------------------------------------------------------

export interface ConstantEvaluator {
  readonly opType: string;
  canEvaluate(node: NodeIR, ctx: FoldingContext): boolean;
  evaluate(node: NodeIR, ctx: FoldingContext): ConstantResult[] | null;
}

export interface ConstantResult {
  name: string;
  dataType: MLOperandDataType;
  shape: number[];
  rawData: Uint8Array;
}

// ---------------------------------------------------------------------------
// FoldingContext — immutable view of known constants + shapes
// ---------------------------------------------------------------------------

export class FoldingContext {
  private constants: Map<string, ConstantInfo>;
  private shapes: Map<string, (number | string)[]>;
  private dataTypes: Map<string, MLOperandDataType>;
  private graphInputNames: Set<string>;
  private producerMap: Map<string, NodeIR>;

  constructor(
    constants: ConstantInfo[],
    shapes?: Map<string, (number | string)[]>,
    dataTypes?: Map<string, MLOperandDataType>,
    graphInputNames?: Set<string>,
    nodes?: NodeIR[],
  ) {
    this.constants = new Map();
    for (const c of constants) this.constants.set(c.name, c);
    this.shapes = shapes ?? new Map();
    this.dataTypes = dataTypes ?? new Map();
    this.graphInputNames = graphInputNames ?? new Set();
    this.producerMap = new Map();
    if (nodes) {
      for (const n of nodes) {
        for (const o of n.outputs) if (o) this.producerMap.set(o, n);
      }
    }
  }

  isConstant(name: string): boolean {
    return this.constants.has(name);
  }

  /** True if name is a graph-level input (not a constant, not an intermediate). */
  isGraphInput(name: string): boolean {
    return this.graphInputNames.has(name);
  }

  getConstant(name: string): ConstantInfo | undefined {
    return this.constants.get(name);
  }

  getShape(name: string): (number | string)[] | undefined {
    return this.shapes.get(name) ?? this.constants.get(name)?.shape;
  }

  getDataType(name: string): MLOperandDataType | undefined {
    return this.dataTypes.get(name) ?? this.constants.get(name)?.dataType;
  }

  addConstant(c: ConstantInfo): void {
    this.constants.set(c.name, c);
    this.shapes.set(c.name, c.shape);
    this.dataTypes.set(c.name, c.dataType);
  }

  /** Read scalar number from a constant tensor. Handles int64, int32, float32, float16. */
  readScalar(name: string): number | null {
    const c = this.constants.get(name);
    if (!c || c.rawData.byteLength === 0) return null;
    return readScalarFromConstant(c);
  }

  /** Read all values from a constant tensor as number[]. */
  readValues(name: string): number[] | null {
    const c = this.constants.get(name);
    if (!c) return null;
    return readValuesFromConstant(c);
  }

  /** Read all values from a constant tensor as bigint[] (for int64). */
  readBigIntValues(name: string): bigint[] | null {
    const c = this.constants.get(name);
    if (!c) return null;
    return readBigIntValuesFromConstant(c);
  }

  /**
   * Resolve a scalar integer value by tracing through shape-computation chains.
   * Handles: constants, Shape → dim lookup, Gather(Shape, idx), Cast(scalar).
   * Used by Range/ConstantOfShape evaluators to resolve inputs that come from
   * Shape chains on graph inputs without requiring those Shape nodes to fold.
   */
  resolveScalar(name: string, depth = 0): number | null {
    if (depth > 20) return null;

    // 1. Already a constant
    const scalar = this.readScalar(name);
    if (scalar != null) return scalar;

    // 2. Trace through producer nodes
    const producer = this.producerMap.get(name);
    if (!producer) return null;

    if (producer.opType === 'Shape') {
      // Shape(X) → returns the full shape as an int64 vector.
      // Only useful when consumed by Gather to pick a single dim.
      // Return null here — Gather will handle Shape → Gather pattern.
      return null;
    }

    if (producer.opType === 'Gather') {
      const dataName = producer.inputs[0];
      const idxName = producer.inputs[1];

      // Resolve index
      const idx = this.resolveScalar(idxName, depth + 1);
      if (idx == null) return null;

      // Check if data comes from Shape node → read from shapes map
      const dataProducer = this.producerMap.get(dataName);
      if (dataProducer?.opType === 'Shape') {
        const inputShape = this.getShape(dataProducer.inputs[0]);
        if (!inputShape || !inputShape.every(d => typeof d === 'number')) return null;
        const dims = inputShape as number[];
        // Handle start/end attributes on Shape (opset 15+)
        const start = (dataProducer.attributes.start as number) ?? 0;
        const end = (dataProducer.attributes.end as number) ?? dims.length;
        const normalStart = start < 0 ? Math.max(0, dims.length + start) : Math.min(start, dims.length);
        const normalEnd = end < 0 ? Math.max(0, dims.length + end) : Math.min(end, dims.length);
        const sliced = dims.slice(normalStart, normalEnd);
        const normalIdx = idx < 0 ? idx + sliced.length : idx;
        if (normalIdx >= 0 && normalIdx < sliced.length) return sliced[normalIdx];
        return null;
      }

      // Data from a constant
      const data = this.readValues(dataName);
      if (!data) return null;
      const normalIdx = idx < 0 ? idx + data.length : idx;
      if (normalIdx >= 0 && normalIdx < data.length) return data[normalIdx];
      return null;
    }

    if (producer.opType === 'Cast') {
      return this.resolveScalar(producer.inputs[0], depth + 1);
    }

    if (producer.opType === 'Add') {
      const a = this.resolveScalar(producer.inputs[0], depth + 1);
      const b = this.resolveScalar(producer.inputs[1], depth + 1);
      if (a != null && b != null) return a + b;
      return null;
    }

    if (producer.opType === 'Sub') {
      const a = this.resolveScalar(producer.inputs[0], depth + 1);
      const b = this.resolveScalar(producer.inputs[1], depth + 1);
      if (a != null && b != null) return a - b;
      return null;
    }

    if (producer.opType === 'Mul') {
      const a = this.resolveScalar(producer.inputs[0], depth + 1);
      const b = this.resolveScalar(producer.inputs[1], depth + 1);
      if (a != null && b != null) return a * b;
      return null;
    }

    if (producer.opType === 'Unsqueeze' || producer.opType === 'Squeeze' ||
        producer.opType === 'Reshape' || producer.opType === 'Identity') {
      return this.resolveScalar(producer.inputs[0], depth + 1);
    }

    return null;
  }

  /**
   * Resolve an integer vector by tracing through shape-computation chains.
   * Handles: constants, Shape → shapes map, Concat of resolvable vectors,
   * Gather(Shape, idx) → single-element vector,  Unsqueeze(scalar) → [scalar].
   * Used by ConstantOfShape to resolve shape inputs from Shape chains.
   */
  resolveIntVector(name: string, depth = 0): number[] | null {
    if (depth > 20) return null;

    // 1. Already a constant
    const vals = this.readValues(name);
    if (vals) return vals.map(Math.trunc);

    // 2. Trace through producer nodes
    const producer = this.producerMap.get(name);
    if (!producer) return null;

    if (producer.opType === 'Shape') {
      const inputShape = this.getShape(producer.inputs[0]);
      if (!inputShape || !inputShape.every(d => typeof d === 'number')) return null;
      const dims = inputShape as number[];
      const start = (producer.attributes.start as number) ?? 0;
      const end = (producer.attributes.end as number) ?? dims.length;
      const normalStart = start < 0 ? Math.max(0, dims.length + start) : Math.min(start, dims.length);
      const normalEnd = end < 0 ? Math.max(0, dims.length + end) : Math.min(end, dims.length);
      return dims.slice(normalStart, normalEnd);
    }

    if (producer.opType === 'Concat') {
      const axis = (producer.attributes.axis as number) ?? 0;
      if (axis !== 0) return null; // Only handle axis=0 concatenation
      const parts: number[] = [];
      for (const inp of producer.inputs) {
        if (!inp) return null;
        const v = this.resolveIntVector(inp, depth + 1);
        if (!v) return null;
        parts.push(...v);
      }
      return parts;
    }

    if (producer.opType === 'Gather') {
      // Gather(data, idx) → single element from data vector
      const scalar = this.resolveScalar(name, depth + 1);
      if (scalar != null) return [scalar];
      return null;
    }

    if (producer.opType === 'Unsqueeze') {
      const scalar = this.resolveScalar(producer.inputs[0], depth + 1);
      if (scalar != null) return [scalar];
      return this.resolveIntVector(producer.inputs[0], depth + 1);
    }

    if (producer.opType === 'Cast' || producer.opType === 'Identity' ||
        producer.opType === 'Squeeze' || producer.opType === 'Reshape') {
      return this.resolveIntVector(producer.inputs[0], depth + 1);
    }

    return null;
  }
}

// ---------------------------------------------------------------------------
// Data reading helpers
// ---------------------------------------------------------------------------

function readScalarFromConstant(c: ConstantInfo): number | null {
  const view = new DataView(c.rawData.buffer, c.rawData.byteOffset, c.rawData.byteLength);
  switch (c.dataType) {
    case 'int64': return Number(view.getBigInt64(0, true));
    case 'uint64': return Number(view.getBigUint64(0, true));
    case 'int32': return view.getInt32(0, true);
    case 'uint32': return view.getUint32(0, true);
    case 'float32': return view.getFloat32(0, true);
    case 'float16': return float16ToFloat32(view.getUint16(0, true));
    case 'int8': return view.getInt8(0);
    case 'uint8': return view.getUint8(0);
    default: return null;
  }
}

function readValuesFromConstant(c: ConstantInfo): number[] | null {
  const bpe = bytesPerElement(c.dataType);
  const count = c.rawData.byteLength / bpe;
  if (count === 0) return [];
  const view = new DataView(c.rawData.buffer, c.rawData.byteOffset, c.rawData.byteLength);
  const result: number[] = [];
  for (let i = 0; i < count; i++) {
    switch (c.dataType) {
      case 'int64': result.push(Number(view.getBigInt64(i * 8, true))); break;
      case 'uint64': result.push(Number(view.getBigUint64(i * 8, true))); break;
      case 'int32': result.push(view.getInt32(i * 4, true)); break;
      case 'uint32': result.push(view.getUint32(i * 4, true)); break;
      case 'float32': result.push(view.getFloat32(i * 4, true)); break;
      case 'float16': result.push(float16ToFloat32(view.getUint16(i * 2, true))); break;
      case 'int8': result.push(view.getInt8(i)); break;
      case 'uint8': result.push(view.getUint8(i)); break;
      default: return null;
    }
  }
  return result;
}

function readBigIntValuesFromConstant(c: ConstantInfo): bigint[] | null {
  const bpe = bytesPerElement(c.dataType);
  const count = c.rawData.byteLength / bpe;
  if (count === 0) return [];
  const view = new DataView(c.rawData.buffer, c.rawData.byteOffset, c.rawData.byteLength);
  const result: bigint[] = [];
  for (let i = 0; i < count; i++) {
    switch (c.dataType) {
      case 'int64': result.push(view.getBigInt64(i * 8, true)); break;
      case 'uint64': result.push(view.getBigUint64(i * 8, true)); break;
      case 'int32': result.push(BigInt(view.getInt32(i * 4, true))); break;
      case 'uint32': result.push(BigInt(view.getUint32(i * 4, true))); break;
      case 'float32': result.push(BigInt(Math.round(view.getFloat32(i * 4, true)))); break;
      case 'int8': result.push(BigInt(view.getInt8(i))); break;
      case 'uint8': result.push(BigInt(view.getUint8(i))); break;
      default: return null;
    }
  }
  return result;
}

/** IEEE 754 half-precision to single-precision. */
function float16ToFloat32(h: number): number {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) {
    // Subnormal
    if (frac === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 0x1f) {
    return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

// ---------------------------------------------------------------------------
// Pack values into raw bytes
// ---------------------------------------------------------------------------

function packInt64(values: number[]): Uint8Array {
  const buf = new ArrayBuffer(values.length * 8);
  const view = new DataView(buf);
  for (let i = 0; i < values.length; i++) {
    view.setBigInt64(i * 8, BigInt(Math.trunc(values[i])), true);
  }
  return new Uint8Array(buf);
}

function packInt32(values: number[]): Uint8Array {
  return new Uint8Array(new Int32Array(values).buffer);
}

function packFloat32(values: number[]): Uint8Array {
  return new Uint8Array(new Float32Array(values).buffer);
}

function packUint8(values: number[]): Uint8Array {
  return new Uint8Array(values);
}

/** IEEE 754 single-precision to half-precision. */
function float32ToFloat16(f: number): number {
  const buf = new ArrayBuffer(4);
  new Float32Array(buf)[0] = f;
  const bits = new Uint32Array(buf)[0];
  const sign = (bits >> 16) & 0x8000;
  const exp = ((bits >> 23) & 0xff) - 127 + 15;
  const frac = (bits >> 13) & 0x3ff;
  if (exp <= 0) {
    // Subnormal or zero
    if (exp < -10) return sign; // too small → ±0
    const m = (0x400 | frac) >> (1 - exp);
    return sign | m;
  }
  if (exp >= 0x1f) {
    // Overflow → Inf or NaN
    return sign | 0x7c00 | (frac ? 0x200 : 0);
  }
  return sign | (exp << 10) | frac;
}

function packFloat16(values: number[]): Uint8Array {
  const buf = new ArrayBuffer(values.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < values.length; i++) {
    view.setUint16(i * 2, float32ToFloat16(values[i]), true);
  }
  return new Uint8Array(buf);
}

function packUint64(values: number[]): Uint8Array {
  const buf = new ArrayBuffer(values.length * 8);
  const view = new DataView(buf);
  for (let i = 0; i < values.length; i++) {
    view.setBigUint64(i * 8, BigInt(Math.trunc(values[i])), true);
  }
  return new Uint8Array(buf);
}

function packValues(values: number[], dataType: MLOperandDataType): Uint8Array {
  switch (dataType) {
    case 'float32': return packFloat32(values);
    case 'float16': return packFloat16(values);
    case 'int64': return packInt64(values);
    case 'uint64': return packUint64(values);
    case 'int32': return packInt32(values);
    case 'uint32': return new Uint8Array(new Uint32Array(values).buffer);
    case 'int8': return new Uint8Array(new Int8Array(values).buffer);
    case 'uint8': return packUint8(values);
  }
}

// ---------------------------------------------------------------------------
// Evaluator implementations
// ---------------------------------------------------------------------------

/** Shape: extract tensor shape as int64 vector.
 *  Only folds when the input is a constant or a graph input — NOT intermediate
 *  runtime tensors. Folding Shape(intermediate) can produce incorrect reshape
 *  targets when the shapes map disagrees with actual runtime tensor shapes
 *  (e.g. data-dependent gathers like ArgMax → Gather). */
const ShapeEvaluator: ConstantEvaluator = {
  opType: 'Shape',
  canEvaluate(node, ctx) {
    const input = node.inputs[0];
    // Only fold Shape on constants or graph inputs, not intermediate runtime tensors
    if (!ctx.isConstant(input) && !ctx.isGraphInput(input)) return false;
    const shape = ctx.getShape(input);
    return !!shape && shape.every(d => typeof d === 'number');
  },
  evaluate(node, ctx) {
    const shape = ctx.getShape(node.inputs[0]);
    if (!shape) return null;
    const dims = shape as number[];
    // Handle start/end attributes (ONNX opset 15+)
    const start = (node.attributes.start as number) ?? 0;
    const end = (node.attributes.end as number) ?? dims.length;
    const normalStart = start < 0 ? Math.max(0, dims.length + start) : Math.min(start, dims.length);
    const normalEnd = end < 0 ? Math.max(0, dims.length + end) : Math.min(end, dims.length);
    const sliced = dims.slice(normalStart, normalEnd);
    return [{
      name: node.outputs[0],
      dataType: 'int64',
      shape: [sliced.length],
      rawData: packInt64(sliced),
    }];
  },
};

/** Gather: index into a constant tensor (axis-0, scalar/1D index) */
const GatherEvaluator: ConstantEvaluator = {
  opType: 'Gather',
  canEvaluate(node, ctx) {
    return ctx.isConstant(node.inputs[0]) && ctx.isConstant(node.inputs[1]);
  },
  evaluate(node, ctx) {
    const data = ctx.readValues(node.inputs[0]);
    const indices = ctx.readValues(node.inputs[1]);
    if (!data || !indices) return null;

    const dataConst = ctx.getConstant(node.inputs[0])!;
    const idxConst = ctx.getConstant(node.inputs[1])!;
    const axis = (node.attributes.axis as number) ?? 0;
    const dataShape = dataConst.shape as number[];
    const normalAxis = axis < 0 ? axis + dataShape.length : axis;

    if (dataShape.length === 0) return null;

    const axisDim = dataShape[normalAxis];
    // Normalize negative indices
    const normalizedIndices = indices.map(idx => idx < 0 ? idx + axisDim : idx);

    // For the common case: 1D data with scalar/1D indices
    if (dataShape.length === 1) {
      const result = normalizedIndices.map(i => data[i]);
      const outShape = idxConst.shape.length === 0 ? [] : [normalizedIndices.length];
      return [{
        name: node.outputs[0],
        dataType: dataConst.dataType,
        shape: outShape as number[],
        rawData: packValues(result, dataConst.dataType),
      }];
    }

    // General case: multi-dimensional data
    // Output shape: data.shape[:axis] + indices.shape + data.shape[axis+1:]
    const idxShape = idxConst.shape as number[];
    const outShape = [
      ...dataShape.slice(0, normalAxis),
      ...(idxShape.length === 0 ? [] : idxShape),
      ...dataShape.slice(normalAxis + 1),
    ];

    // Compute stride for the gather axis
    const innerSize = dataShape.slice(normalAxis + 1).reduce((a, b) => a * b, 1);
    const outerSize = dataShape.slice(0, normalAxis).reduce((a, b) => a * b, 1);

    const result: number[] = [];
    for (let outer = 0; outer < outerSize; outer++) {
      for (const idx of normalizedIndices) {
        const baseOffset = outer * axisDim * innerSize + idx * innerSize;
        for (let inner = 0; inner < innerSize; inner++) {
          result.push(data[baseOffset + inner]);
        }
      }
    }

    return [{
      name: node.outputs[0],
      dataType: dataConst.dataType,
      shape: outShape,
      rawData: packValues(result, dataConst.dataType),
    }];
  },
};

/** Concat: concatenate constant tensors along axis */
const ConcatEvaluator: ConstantEvaluator = {
  opType: 'Concat',
  canEvaluate(node, ctx) {
    return node.inputs.every(inp => !inp || ctx.isConstant(inp));
  },
  evaluate(node, ctx) {
    const axis = (node.attributes.axis as number) ?? 0;
    const inputs = node.inputs.filter(Boolean);
    if (inputs.length === 0) return null;

    const firstConst = ctx.getConstant(inputs[0])!;
    const firstShape = firstConst.shape as number[];
    const normalAxis = axis < 0 ? axis + firstShape.length : axis;

    // For 1D tensors (common: shape computation chains), just concatenate values
    if (firstShape.length <= 1) {
      const allValues: number[] = [];
      for (const inp of inputs) {
        const vals = ctx.readValues(inp);
        if (!vals) return null;
        allValues.push(...vals);
      }
      return [{
        name: node.outputs[0],
        dataType: firstConst.dataType,
        shape: [allValues.length],
        rawData: packValues(allValues, firstConst.dataType),
      }];
    }

    // General N-D case: concatenate along axis
    // Validate all inputs have same shape except at concat axis
    const allShapes = inputs.map(inp => (ctx.getConstant(inp)!.shape as number[]));
    const outShape = [...firstShape];
    let totalAxisDim = 0;
    for (const s of allShapes) {
      totalAxisDim += s[normalAxis];
    }
    outShape[normalAxis] = totalAxisDim;

    // Simple approach: read all values, interleave at the correct axis
    // For N-D, compute using slices
    const allData = inputs.map(inp => ctx.readValues(inp)!);
    if (allData.some(d => !d)) return null;

    const result: number[] = [];
    const outerSize = outShape.slice(0, normalAxis).reduce((a, b) => a * b, 1);

    for (let outer = 0; outer < outerSize; outer++) {
      for (let t = 0; t < inputs.length; t++) {
        const tShape = allShapes[t];
        const tAxisDim = tShape[normalAxis];
        const innerSize = tShape.slice(normalAxis + 1).reduce((a, b) => a * b, 1);
        const tTotalInner = tAxisDim * innerSize;
        const offset = outer * tTotalInner;
        for (let i = 0; i < tTotalInner; i++) {
          result.push(allData[t][offset + i]);
        }
      }
    }

    return [{
      name: node.outputs[0],
      dataType: firstConst.dataType,
      shape: outShape,
      rawData: packValues(result, firstConst.dataType),
    }];
  },
};

/** Unsqueeze: insert dimensions of size 1 */
const UnsqueezeEvaluator: ConstantEvaluator = {
  opType: 'Unsqueeze',
  canEvaluate(node, ctx) {
    if (!ctx.isConstant(node.inputs[0])) return false;
    // Axes from attribute (opset < 13) or second input (opset 13+)
    const axesAttr = node.attributes.axes as number[] | undefined;
    if (axesAttr) return true;
    return node.inputs.length > 1 && !!node.inputs[1] && ctx.isConstant(node.inputs[1]);
  },
  evaluate(node, ctx) {
    const inputConst = ctx.getConstant(node.inputs[0])!;
    const inputShape = inputConst.shape as number[];

    let axes: number[];
    const axesAttr = node.attributes.axes as number[] | undefined;
    if (axesAttr) {
      axes = axesAttr;
    } else {
      const axesVals = ctx.readValues(node.inputs[1]);
      if (!axesVals) return null;
      axes = axesVals;
    }

    const expandedRank = inputShape.length + axes.length;
    const resolved = axes.map(a => a < 0 ? a + expandedRank : a).sort((a, b) => a - b);
    const outShape: number[] = [];
    let srcIdx = 0;
    for (let i = 0; i < expandedRank; i++) {
      if (resolved.includes(i)) {
        outShape.push(1);
      } else {
        outShape.push(inputShape[srcIdx++]);
      }
    }

    // Data is unchanged, only shape changes
    return [{
      name: node.outputs[0],
      dataType: inputConst.dataType,
      shape: outShape,
      rawData: new Uint8Array(inputConst.rawData),
    }];
  },
};

/** Squeeze: remove dimensions of size 1 */
const SqueezeEvaluator: ConstantEvaluator = {
  opType: 'Squeeze',
  canEvaluate(node, ctx) {
    if (!ctx.isConstant(node.inputs[0])) return false;
    const axesAttr = node.attributes.axes as number[] | undefined;
    if (axesAttr) return true;
    if (node.inputs.length > 1 && node.inputs[1]) return ctx.isConstant(node.inputs[1]);
    return true; // No axes specified = squeeze all size-1 dims
  },
  evaluate(node, ctx) {
    const inputConst = ctx.getConstant(node.inputs[0])!;
    const inputShape = inputConst.shape as number[];

    let axes: number[] | undefined;
    const axesAttr = node.attributes.axes as number[] | undefined;
    if (axesAttr) {
      axes = axesAttr;
    } else if (node.inputs.length > 1 && node.inputs[1]) {
      const axesVals = ctx.readValues(node.inputs[1]);
      if (axesVals) axes = axesVals;
    }

    let outShape: number[];
    if (axes && axes.length > 0) {
      const resolved = new Set(axes.map(a => a < 0 ? a + inputShape.length : a));
      outShape = inputShape.filter((_, i) => !resolved.has(i));
    } else {
      outShape = inputShape.filter(d => d !== 1);
    }

    return [{
      name: node.outputs[0],
      dataType: inputConst.dataType,
      shape: outShape,
      rawData: new Uint8Array(inputConst.rawData),
    }];
  },
};

/** Cast: type conversion between constant tensors */
const CastEvaluator: ConstantEvaluator = {
  opType: 'Cast',
  canEvaluate(node, ctx) {
    return ctx.isConstant(node.inputs[0]) && node.attributes.to != null;
  },
  evaluate(node, ctx) {
    const inputConst = ctx.getConstant(node.inputs[0])!;
    const values = ctx.readValues(node.inputs[0]);
    if (!values) return null;

    const toOnnx = node.attributes.to as number;
    // Map ONNX data type ID to WebNN type
    const targetType = onnxDataTypeFromId(toOnnx);
    if (!targetType) return null;

    return [{
      name: node.outputs[0],
      dataType: targetType,
      shape: inputConst.shape as number[],
      rawData: packValues(values, targetType),
    }];
  },
};

/** Range: generate a sequence (start, limit, delta).
 *  Resolves inputs via constants OR shape-computation chains (Shape → Gather).
 *  This allows Range to fold even when Shape nodes don't produce constants. */
const RangeEvaluator: ConstantEvaluator = {
  opType: 'Range',
  canEvaluate(node, ctx) {
    if (node.inputs.length !== 3) return false;
    // Each input must be resolvable to a scalar (constant or via shape chain)
    for (const inp of node.inputs) {
      if (!ctx.isConstant(inp) && ctx.resolveScalar(inp) == null) return false;
    }
    return true;
  },
  evaluate(node, ctx) {
    const start = ctx.readScalar(node.inputs[0]) ?? ctx.resolveScalar(node.inputs[0]);
    const limit = ctx.readScalar(node.inputs[1]) ?? ctx.resolveScalar(node.inputs[1]);
    const delta = ctx.readScalar(node.inputs[2]) ?? ctx.resolveScalar(node.inputs[2]);
    if (start == null || limit == null || delta == null || delta === 0) return null;

    const values: number[] = [];
    if (delta > 0) {
      for (let v = start; v < limit; v += delta) values.push(v);
    } else {
      for (let v = start; v > limit; v += delta) values.push(v);
    }

    // Output type: prefer constant's declared type, fall back to int64
    const startConst = ctx.getConstant(node.inputs[0]);
    const dataType = startConst?.dataType ?? ctx.getDataType(node.inputs[0]) ?? 'int64';
    return [{
      name: node.outputs[0],
      dataType,
      shape: [values.length],
      rawData: packValues(values, dataType),
    }];
  },
};

/** ConstantOfShape: create a tensor filled with a value.
 *  Resolves shape input via constants OR shape-computation chains. */
const ConstantOfShapeEvaluator: ConstantEvaluator = {
  opType: 'ConstantOfShape',
  canEvaluate(node, ctx) {
    // Shape input must be resolvable (constant or via shape chain)
    return ctx.isConstant(node.inputs[0]) || ctx.resolveIntVector(node.inputs[0]) != null;
  },
  evaluate(node, ctx) {
    const shapeVals = ctx.readValues(node.inputs[0]) ?? ctx.resolveIntVector(node.inputs[0]);
    if (!shapeVals) return null;
    const shape = shapeVals.map(Math.trunc);

    // Default fill value is 0.0 float32; can be overridden by 'value' attribute
    let fillValue = 0;
    let dataType: MLOperandDataType = 'float32';

    const valueAttr = node.attributes.value;
    if (valueAttr != null && typeof valueAttr === 'object') {
      // The value attribute is a tensor with a single element
      const v = valueAttr as { floatData?: number[]; int32Data?: number[]; int64Data?: (number | bigint)[]; rawData?: Uint8Array; dataType?: number };
      if (v.floatData && v.floatData.length > 0) {
        fillValue = v.floatData[0];
        dataType = 'float32';
      } else if (v.int32Data && v.int32Data.length > 0) {
        fillValue = v.int32Data[0];
        dataType = 'int32';
      } else if (v.int64Data && v.int64Data.length > 0) {
        fillValue = Number(v.int64Data[0]);
        dataType = 'int64';
      } else if (v.rawData && v.rawData.byteLength > 0) {
        const dt = v.dataType ?? 1;
        dataType = onnxDataTypeFromId(dt) ?? 'float32';
        const view = new DataView(v.rawData.buffer, v.rawData.byteOffset, v.rawData.byteLength);
        switch (dataType) {
          case 'float32': fillValue = view.getFloat32(0, true); break;
          case 'int32': fillValue = view.getInt32(0, true); break;
          case 'int64': fillValue = Number(view.getBigInt64(0, true)); break;
          case 'float16': fillValue = float16ToFloat32(view.getUint16(0, true)); break;
          case 'int8': fillValue = view.getInt8(0); break;
          case 'uint8': fillValue = view.getUint8(0); break;
          default: fillValue = view.getFloat32(0, true); break;
        }
      }
    }

    const totalElements = shape.reduce((a, b) => a * b, 1);
    const values = new Array(totalElements).fill(fillValue);

    return [{
      name: node.outputs[0],
      dataType,
      shape,
      rawData: packValues(values, dataType),
    }];
  },
};

/** Reshape: reshapes a constant (data unchanged, shape updated) */
const ReshapeEvaluator: ConstantEvaluator = {
  opType: 'Reshape',
  canEvaluate(node, ctx) {
    return ctx.isConstant(node.inputs[0]) &&
      node.inputs.length > 1 && !!node.inputs[1] && ctx.isConstant(node.inputs[1]);
  },
  evaluate(node, ctx) {
    const inputConst = ctx.getConstant(node.inputs[0])!;
    const shapeVals = ctx.readValues(node.inputs[1]);
    if (!shapeVals) return null;

    const inputShape = inputConst.shape as number[];
    const allowZero = (node.attributes.allowzero as number) ?? 0;
    const targetShape = shapeVals.map(Math.trunc);

    // Resolve 0 and -1 special values
    const resolved = [...targetShape];
    if (!allowZero) {
      for (let i = 0; i < resolved.length; i++) {
        if (resolved[i] === 0 && i < inputShape.length) resolved[i] = inputShape[i];
      }
    }
    const inferIdx = resolved.indexOf(-1);
    if (inferIdx !== -1) {
      const totalInput = inputShape.reduce((a, b) => a * b, 1);
      const knownProduct = resolved.reduce((a, b, i) => i === inferIdx ? a : a * b, 1);
      if (knownProduct <= 0) return null;
      resolved[inferIdx] = totalInput / knownProduct;
    }
    if (resolved.some(d => d < 0 || !Number.isInteger(d))) return null;

    return [{
      name: node.outputs[0],
      dataType: inputConst.dataType,
      shape: resolved,
      rawData: new Uint8Array(inputConst.rawData),
    }];
  },
};

/** Slice: extract a sub-tensor */
const SliceEvaluator: ConstantEvaluator = {
  opType: 'Slice',
  canEvaluate(node, ctx) {
    // data, starts, ends must be constant; axes and steps are optional
    if (!ctx.isConstant(node.inputs[0])) return false;
    if (!ctx.isConstant(node.inputs[1])) return false;
    if (!ctx.isConstant(node.inputs[2])) return false;
    if (node.inputs[3] && !ctx.isConstant(node.inputs[3])) return false;
    if (node.inputs[4] && !ctx.isConstant(node.inputs[4])) return false;
    return true;
  },
  evaluate(node, ctx) {
    const dataConst = ctx.getConstant(node.inputs[0])!;
    const dataShape = dataConst.shape as number[];
    const data = ctx.readValues(node.inputs[0]);
    const starts = ctx.readValues(node.inputs[1]);
    const ends = ctx.readValues(node.inputs[2]);
    if (!data || !starts || !ends) return null;

    const rank = dataShape.length;

    // Handle 1D case (most common for shape computation chains)
    if (rank <= 1) {
      const len = data.length;
      const axesVals = node.inputs[3] ? ctx.readValues(node.inputs[3]) : null;
      const stepsVals = node.inputs[4] ? ctx.readValues(node.inputs[4]) : null;
      const axis = axesVals ? axesVals[0] : 0;
      if (axis !== 0) return null; // 1D only has axis 0
      const step = stepsVals ? stepsVals[0] : 1;
      const start = clampIndex(starts[0], len);
      const end = clampIndex(ends[0], len);

      const result: number[] = [];
      if (step > 0) {
        for (let i = start; i < end; i += step) result.push(data[i]);
      } else {
        for (let i = start; i > end; i += step) result.push(data[i]);
      }
      return [{
        name: node.outputs[0],
        dataType: dataConst.dataType,
        shape: [result.length],
        rawData: packValues(result, dataConst.dataType),
      }];
    }

    // General N-D case
    const axes = node.inputs[3] ? ctx.readValues(node.inputs[3])?.map(a => a < 0 ? a + rank : a) : Array.from({ length: starts.length }, (_, i) => i);
    const steps = node.inputs[4] ? ctx.readValues(node.inputs[4]) : new Array(starts.length).fill(1);
    if (!axes || !steps) return null;

    // Compute output shape
    const outShape = [...dataShape];
    const sliceParams: { start: number; end: number; step: number }[] = [];
    for (let i = 0; i < rank; i++) {
      const axisIdx = axes.indexOf(i);
      if (axisIdx === -1) {
        sliceParams.push({ start: 0, end: dataShape[i], step: 1 });
      } else {
        const step = steps[axisIdx];
        const start = clampIndex(starts[axisIdx], dataShape[i]);
        const end = clampIndex(ends[axisIdx], dataShape[i]);
        let count = 0;
        if (step > 0) {
          for (let j = start; j < end; j += step) count++;
        } else {
          for (let j = start; j > end; j += step) count++;
        }
        outShape[i] = count;
        sliceParams.push({ start, end, step });
      }
    }

    const totalOutput = outShape.reduce((a, b) => a * b, 1);
    const result = new Array(totalOutput);
    const strides = computeStrides(dataShape);
    const outStrides = computeStrides(outShape);

    for (let outIdx = 0; outIdx < totalOutput; outIdx++) {
      // Convert flat output index to N-D coords
      const outCoords = flatToNd(outIdx, outStrides);
      // Map to input coords
      const inCoords = outCoords.map((c, dim) => {
        const p = sliceParams[dim];
        return p.start + c * p.step;
      });
      // Convert input N-D coords to flat index
      const inIdx = ndToFlat(inCoords, strides);
      result[outIdx] = data[inIdx];
    }

    return [{
      name: node.outputs[0],
      dataType: dataConst.dataType,
      shape: outShape,
      rawData: packValues(result, dataConst.dataType),
    }];
  },
};

/** Elementwise binary ops on constants: Add, Sub, Mul, Div, Pow, Equal, Greater, Less, etc. */
function makeElementwiseEvaluator(
  opType: string,
  fn: (a: number, b: number) => number,
  outputType?: MLOperandDataType,
): ConstantEvaluator {
  return {
    opType,
    canEvaluate(node, ctx) {
      return node.inputs.every(inp => !inp || ctx.isConstant(inp));
    },
    evaluate(node, ctx) {
      const aConst = ctx.getConstant(node.inputs[0])!;
      const bConst = ctx.getConstant(node.inputs[1])!;
      const aVals = ctx.readValues(node.inputs[0]);
      const bVals = ctx.readValues(node.inputs[1]);
      if (!aVals || !bVals) return null;

      const aShape = aConst.shape as number[];
      const bShape = bConst.shape as number[];
      const outShape = broadcastShapes(aShape, bShape);
      if (!outShape) return null;

      const totalOutput = outShape.reduce((a, b) => a * b, 1);
      const result = new Array(totalOutput);

      const aBroadcast = computeBroadcastStrides(aShape, outShape);
      const bBroadcast = computeBroadcastStrides(bShape, outShape);
      const outStrides = computeStrides(outShape);

      for (let i = 0; i < totalOutput; i++) {
        const coords = flatToNd(i, outStrides);
        const aIdx = broadcastIndex(coords, aBroadcast);
        const bIdx = broadcastIndex(coords, bBroadcast);
        result[i] = fn(aVals[aIdx], bVals[bIdx]);
      }

      const outType = outputType ?? aConst.dataType;
      return [{
        name: node.outputs[0],
        dataType: outType,
        shape: outShape,
        rawData: packValues(result, outType),
      }];
    },
  };
}

/** Where: condition ? x : y */
const WhereEvaluator: ConstantEvaluator = {
  opType: 'Where',
  canEvaluate(node, ctx) {
    return node.inputs.length === 3 &&
      ctx.isConstant(node.inputs[0]) &&
      ctx.isConstant(node.inputs[1]) &&
      ctx.isConstant(node.inputs[2]);
  },
  evaluate(node, ctx) {
    const condConst = ctx.getConstant(node.inputs[0])!;
    const xConst = ctx.getConstant(node.inputs[1])!;
    const yConst = ctx.getConstant(node.inputs[2])!;
    const condVals = ctx.readValues(node.inputs[0]);
    const xVals = ctx.readValues(node.inputs[1]);
    const yVals = ctx.readValues(node.inputs[2]);
    if (!condVals || !xVals || !yVals) return null;

    const condShape = condConst.shape as number[];
    const xShape = xConst.shape as number[];
    const yShape = yConst.shape as number[];

    // Broadcast all three shapes together
    const xyShape = broadcastShapes(xShape, yShape);
    if (!xyShape) return null;
    const outShape = broadcastShapes(condShape, xyShape);
    if (!outShape) return null;

    const totalOutput = outShape.reduce((a, b) => a * b, 1);
    const result = new Array(totalOutput);

    const condBroadcast = computeBroadcastStrides(condShape, outShape);
    const xBroadcast = computeBroadcastStrides(xShape, outShape);
    const yBroadcast = computeBroadcastStrides(yShape, outShape);
    const outStrides = computeStrides(outShape);

    for (let i = 0; i < totalOutput; i++) {
      const coords = flatToNd(i, outStrides);
      const cIdx = broadcastIndex(coords, condBroadcast);
      const xIdx = broadcastIndex(coords, xBroadcast);
      const yIdx = broadcastIndex(coords, yBroadcast);
      result[i] = condVals[cIdx] ? xVals[xIdx] : yVals[yIdx];
    }

    return [{
      name: node.outputs[0],
      dataType: xConst.dataType,
      shape: outShape,
      rawData: packValues(result, xConst.dataType),
    }];
  },
};

/** Expand: broadcast a tensor to a specified shape */
const ExpandEvaluator: ConstantEvaluator = {
  opType: 'Expand',
  canEvaluate(node, ctx) {
    return ctx.isConstant(node.inputs[0]) &&
      node.inputs.length > 1 && !!node.inputs[1] && ctx.isConstant(node.inputs[1]);
  },
  evaluate(node, ctx) {
    const inputConst = ctx.getConstant(node.inputs[0])!;
    const shapeVals = ctx.readValues(node.inputs[1]);
    const inputVals = ctx.readValues(node.inputs[0]);
    if (!shapeVals || !inputVals) return null;

    const inputShape = inputConst.shape as number[];
    const targetShape = shapeVals.map(Math.trunc);

    // Compute output shape via broadcasting rules
    const outShape = broadcastShapes(inputShape, targetShape);
    if (!outShape) return null;

    const totalOutput = outShape.reduce((a, b) => a * b, 1);
    const result = new Array(totalOutput);
    const inputBroadcast = computeBroadcastStrides(inputShape, outShape);
    const outStrides = computeStrides(outShape);

    for (let i = 0; i < totalOutput; i++) {
      const coords = flatToNd(i, outStrides);
      const inIdx = broadcastIndex(coords, inputBroadcast);
      result[i] = inputVals[inIdx];
    }

    return [{
      name: node.outputs[0],
      dataType: inputConst.dataType,
      shape: outShape,
      rawData: packValues(result, inputConst.dataType),
    }];
  },
};

/** Neg: negate a constant */
const NegEvaluator: ConstantEvaluator = {
  opType: 'Neg',
  canEvaluate(node, ctx) {
    return ctx.isConstant(node.inputs[0]);
  },
  evaluate(node, ctx) {
    const inputConst = ctx.getConstant(node.inputs[0])!;
    const values = ctx.readValues(node.inputs[0]);
    if (!values) return null;
    return [{
      name: node.outputs[0],
      dataType: inputConst.dataType,
      shape: inputConst.shape as number[],
      rawData: packValues(values.map(v => -v), inputConst.dataType),
    }];
  },
};

/** Identity/Flatten/Dropout: pass-through ops */
function makePassthroughEvaluator(opType: string): ConstantEvaluator {
  return {
    opType,
    canEvaluate(node, ctx) {
      return ctx.isConstant(node.inputs[0]);
    },
    evaluate(node, ctx) {
      const inputConst = ctx.getConstant(node.inputs[0])!;
      return [{
        name: node.outputs[0],
        dataType: inputConst.dataType,
        shape: inputConst.shape as number[],
        rawData: new Uint8Array(inputConst.rawData),
      }];
    },
  };
}

/** Transpose: permute dimensions of a constant tensor */
const TransposeEvaluator: ConstantEvaluator = {
  opType: 'Transpose',
  canEvaluate(node, ctx) {
    return ctx.isConstant(node.inputs[0]);
  },
  evaluate(node, ctx) {
    const inputConst = ctx.getConstant(node.inputs[0])!;
    const inputShape = inputConst.shape as number[];
    const values = ctx.readValues(node.inputs[0]);
    if (!values) return null;

    const perm = (node.attributes.perm as number[]) ?? [...Array(inputShape.length).keys()].reverse();
    const outShape = perm.map(i => inputShape[i]);

    const totalElements = values.length;
    const result = new Array(totalElements);
    const inStrides = computeStrides(inputShape);
    const outStrides = computeStrides(outShape);

    for (let outIdx = 0; outIdx < totalElements; outIdx++) {
      const outCoords = flatToNd(outIdx, outStrides);
      const inCoords = new Array(inputShape.length);
      for (let i = 0; i < perm.length; i++) {
        inCoords[perm[i]] = outCoords[i];
      }
      const inIdx = ndToFlat(inCoords, inStrides);
      result[outIdx] = values[inIdx];
    }

    return [{
      name: node.outputs[0],
      dataType: inputConst.dataType,
      shape: outShape,
      rawData: packValues(result, inputConst.dataType),
    }];
  },
};

// ---------------------------------------------------------------------------
// N-D array index helpers
// ---------------------------------------------------------------------------

function computeStrides(shape: number[]): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

function flatToNd(flatIdx: number, strides: number[]): number[] {
  const coords = new Array(strides.length);
  let remaining = flatIdx;
  for (let i = 0; i < strides.length; i++) {
    coords[i] = Math.floor(remaining / strides[i]);
    remaining %= strides[i];
  }
  return coords;
}

function ndToFlat(coords: number[], strides: number[]): number {
  let idx = 0;
  for (let i = 0; i < coords.length; i++) idx += coords[i] * strides[i];
  return idx;
}

function clampIndex(idx: number, dimSize: number): number {
  if (idx < 0) idx += dimSize;
  return Math.max(0, Math.min(idx, dimSize));
}

function broadcastShapes(a: number[], b: number[]): number[] | null {
  const maxRank = Math.max(a.length, b.length);
  const padA = new Array(maxRank - a.length).fill(1).concat(a);
  const padB = new Array(maxRank - b.length).fill(1).concat(b);
  const result: number[] = [];
  for (let i = 0; i < maxRank; i++) {
    if (padA[i] === padB[i]) {
      result.push(padA[i]);
    } else if (padA[i] === 1) {
      result.push(padB[i]);
    } else if (padB[i] === 1) {
      result.push(padA[i]);
    } else {
      return null; // incompatible
    }
  }
  return result;
}

/** Compute broadcast mapping: for each dim in outShape, the stride in the input (0 if broadcast). */
function computeBroadcastStrides(inputShape: number[], outShape: number[]): { strides: number[]; offset: number } {
  const padLen = outShape.length - inputShape.length;
  const inStrides = computeStrides(inputShape);
  const strides = new Array(outShape.length).fill(0);
  for (let i = 0; i < inputShape.length; i++) {
    if (inputShape[i] === outShape[padLen + i]) {
      strides[padLen + i] = inStrides[i];
    }
    // else: broadcast dim, stride stays 0
  }
  return { strides, offset: 0 };
}

function broadcastIndex(coords: number[], broadcast: { strides: number[] }): number {
  let idx = 0;
  for (let i = 0; i < coords.length; i++) {
    idx += coords[i] * broadcast.strides[i];
  }
  return idx;
}

// ---------------------------------------------------------------------------
// ONNX data type mapping (ID → WebNN type)
// ---------------------------------------------------------------------------

function onnxDataTypeFromId(typeId: number): MLOperandDataType | null {
  switch (typeId) {
    case 1: return 'float32';
    case 2: return 'uint8';
    case 3: return 'int8';
    case 5: return 'int32';  // INT16 → int32
    case 6: return 'int32';
    case 7: return 'int64';
    case 9: return 'uint8';  // BOOL → uint8
    case 10: return 'float16';
    case 11: return 'float32'; // DOUBLE → float32
    case 12: return 'uint32';
    case 13: return 'uint64';
    default: return null;
  }
}

// ---------------------------------------------------------------------------
// All evaluators
// ---------------------------------------------------------------------------

const ALL_EVALUATORS: ConstantEvaluator[] = [
  ShapeEvaluator,
  GatherEvaluator,
  ConcatEvaluator,
  UnsqueezeEvaluator,
  SqueezeEvaluator,
  CastEvaluator,
  RangeEvaluator,
  ConstantOfShapeEvaluator,
  ReshapeEvaluator,
  SliceEvaluator,
  makeElementwiseEvaluator('Add', (a, b) => a + b),
  makeElementwiseEvaluator('Sub', (a, b) => a - b),
  makeElementwiseEvaluator('Mul', (a, b) => a * b),
  makeElementwiseEvaluator('Div', (a, b) => b !== 0 ? a / b : 0),
  makeElementwiseEvaluator('Pow', (a, b) => Math.pow(a, b)),
  makeElementwiseEvaluator('Equal', (a, b) => a === b ? 1 : 0, 'uint8'),
  makeElementwiseEvaluator('Greater', (a, b) => a > b ? 1 : 0, 'uint8'),
  makeElementwiseEvaluator('GreaterOrEqual', (a, b) => a >= b ? 1 : 0, 'uint8'),
  makeElementwiseEvaluator('Less', (a, b) => a < b ? 1 : 0, 'uint8'),
  makeElementwiseEvaluator('LessOrEqual', (a, b) => a <= b ? 1 : 0, 'uint8'),
  NegEvaluator,
  WhereEvaluator,
  ExpandEvaluator,
  TransposeEvaluator,
  makePassthroughEvaluator('Identity'),
  makePassthroughEvaluator('Dropout'),
];

// Build lookup for fast dispatch
const evaluatorMap = new Map<string, ConstantEvaluator>();
for (const e of ALL_EVALUATORS) evaluatorMap.set(e.opType, e);

// ---------------------------------------------------------------------------
// Multi-pass constant folding pipeline
// ---------------------------------------------------------------------------

export interface FoldingResult {
  /** Number of nodes folded (removed from graph) */
  nodesFolded: number;
  /** Number of new constants created */
  constantsCreated: number;
  /** Nodes removed per pass (for debugging) */
  passDetails: { pass: number; folded: number }[];
}

/**
 * Run constant folding on a GraphIR. Modifies the graph in-place:
 * - Removes foldable nodes from graph.nodes
 * - Adds computed constants to graph.constants
 * - Updates graph.shapes and graph.dataTypes
 *
 * Runs up to maxIterations passes. Each pass evaluates all nodes whose inputs
 * are all constants. Stops early when a pass folds 0 nodes.
 */
export function foldConstants(graph: GraphIR, maxIterations = 10): FoldingResult {
  const result: FoldingResult = { nodesFolded: 0, constantsCreated: 0, passDetails: [] };

  const graphInputNames = new Set(graph.inputs.map(i => i.name));

  for (let pass = 0; pass < maxIterations; pass++) {
    const ctx = new FoldingContext(
      graph.constants,
      graph.shapes,
      graph.dataTypes,
      graphInputNames,
      graph.nodes,
    );

    const foldedIndices = new Set<number>();
    let foldedThisPass = 0;

    for (let i = 0; i < graph.nodes.length; i++) {
      const node = graph.nodes[i];
      const evaluator = evaluatorMap.get(node.opType);
      if (!evaluator) continue;
      if (!evaluator.canEvaluate(node, ctx)) continue;

      try {
        const results = evaluator.evaluate(node, ctx);
        if (!results || results.length === 0) continue;

        // Add results as new constants
        for (const r of results) {
          const newConst: ConstantInfo = {
            name: r.name,
            dataType: r.dataType,
            shape: r.shape,
            rawData: r.rawData,
            byteLength: r.rawData.byteLength,
          };
          graph.constants.push(newConst);
          ctx.addConstant(newConst);
          if (graph.shapes) graph.shapes.set(r.name, r.shape);
          if (graph.dataTypes) graph.dataTypes.set(r.name, r.dataType);
          result.constantsCreated++;
        }

        foldedIndices.add(i);
        foldedThisPass++;
      } catch {
        // Evaluator failed on this node — skip it, continue with others
        continue;
      }
    }

    result.passDetails.push({ pass: pass + 1, folded: foldedThisPass });
    result.nodesFolded += foldedThisPass;

    if (foldedThisPass === 0) break;

    // Remove folded nodes (iterate in reverse to keep indices stable)
    const sortedIndices = [...foldedIndices].sort((a, b) => b - a);
    for (const idx of sortedIndices) {
      graph.nodes.splice(idx, 1);
    }
  }

  return result;
}
