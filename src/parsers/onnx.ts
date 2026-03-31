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
      shapes.set(name, extractShape(vi));
      dataTypes.set(name, extractDataType(vi));
    }
  }

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
