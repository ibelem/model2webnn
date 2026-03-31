// Intermediate Representation — format-agnostic graph types
// All parsers (ONNX, TFLite, etc.) produce this same structure.

export interface GraphIR {
  name: string;
  format: string; // "onnx" | "tflite"
  inputs: TensorInfo[];
  outputs: TensorInfo[];
  constants: ConstantInfo[];
  nodes: NodeIR[];
  // Optional shape map for all tensors (inputs, outputs, intermediates)
  // Populated from ONNX value_info, TFLite tensor table, etc.
  shapes?: Map<string, (number | string)[]>;
  // Optional data type map for all tensors
  dataTypes?: Map<string, MLOperandDataType>;
}

export interface TensorInfo {
  name: string;
  dataType: MLOperandDataType; // WebNN data types
  shape: (number | string)[]; // string for dynamic dims
}

export interface ConstantInfo extends TensorInfo {
  rawData: Uint8Array;
  byteOffset?: number; // filled during weight packing
  byteLength: number;
}

export interface NodeIR {
  opType: string; // original model op name (e.g. "Conv", "CONV_2D")
  inputs: string[]; // tensor names
  outputs: string[]; // tensor names
  attributes: Record<string, unknown>;
}

// WebNN MLOperandDataType — matches the spec
export type MLOperandDataType =
  | 'float32'
  | 'float16'
  | 'int64'
  | 'uint64'
  | 'int32'
  | 'uint32'
  | 'int8'
  | 'uint8';

// Chromium-only — not in published spec, but required for current Chrome builds
export type MLDeviceType = 'cpu' | 'gpu' | 'npu';
export type MLPowerPreference = 'default' | 'high-performance' | 'low-power';

// MLOperandDataType → TypedArray constructor name
export function getTypedArrayName(dataType: MLOperandDataType): string {
  switch (dataType) {
    case 'float32': return 'Float32Array';
    case 'float16': return 'Float16Array';
    case 'int64': return 'BigInt64Array';
    case 'uint64': return 'BigUint64Array';
    case 'int32': return 'Int32Array';
    case 'uint32': return 'Uint32Array';
    case 'int8': return 'Int8Array';
    case 'uint8': return 'Uint8Array';
  }
}

// Bytes per element for each data type
export function bytesPerElement(dataType: MLOperandDataType): number {
  switch (dataType) {
    case 'float32': return 4;
    case 'float16': return 2;
    case 'int64': return 8;
    case 'uint64': return 8;
    case 'int32': return 4;
    case 'uint32': return 4;
    case 'int8': return 1;
    case 'uint8': return 1;
  }
}

// ONNX TensorProto.DataType → WebNN MLOperandDataType
export function onnxDataType(typeId: number): MLOperandDataType {
  switch (typeId) {
    case 1: return 'float32';   // FLOAT
    case 2: return 'uint8';     // UINT8
    case 3: return 'int8';      // INT8
    case 4: return 'uint32';    // UINT16 → cast to uint32 (WebNN doesn't support uint16)
    case 5: return 'int32';     // INT16 → cast to int32 (WebNN doesn't support int16)
    case 6: return 'int32';     // INT32
    case 7: return 'int64';     // INT64
    case 9: return 'uint8';     // BOOL → cast to uint8
    case 10: return 'float16';  // FLOAT16
    case 11: return 'float32';  // DOUBLE → downcast to float32
    case 12: return 'uint32';   // UINT32
    case 13: return 'uint64';   // UINT64
    case 16: return 'float16';  // BFLOAT16 → cast to float16
    default:
      throw new Error(`Unsupported ONNX data type: ${typeId}`);
  }
}

// TFLite TensorType → WebNN MLOperandDataType
export function tfliteDataType(typeName: string): MLOperandDataType {
  switch (typeName) {
    case 'FLOAT32': return 'float32';
    case 'FLOAT16': return 'float16';
    case 'FLOAT64': return 'float32';  // FLOAT64 → downcast to float32
    case 'INT32': return 'int32';
    case 'UINT32': return 'uint32';
    case 'UINT8': return 'uint8';
    case 'INT8': return 'int8';
    case 'INT16': return 'int32';      // INT16 → cast to int32
    case 'UINT16': return 'uint32';    // UINT16 → cast to uint32
    case 'INT64': return 'int64';
    case 'BOOL': return 'uint8';       // BOOL → cast to uint8
    default:
      throw new Error(`Unsupported TFLite data type: ${typeName}`);
  }
}

/**
 * Collect all unique symbolic (string) dimension names from graph inputs and outputs.
 */
export function getFreeDimensions(graph: GraphIR): string[] {
  const dims = new Set<string>();
  for (const t of [...graph.inputs, ...graph.outputs]) {
    for (const d of t.shape) {
      if (typeof d === 'string') dims.add(d);
    }
  }
  return Array.from(dims).sort();
}

/**
 * Apply freeDimensionOverrides: replace symbolic dimension names with concrete numbers.
 * Modifies shapes in-place on inputs, outputs, and the shapes map.
 */
export function applyFreeDimensionOverrides(
  graph: GraphIR,
  overrides: Record<string, number>,
): void {
  function resolveShape(shape: (number | string)[]): (number | string)[] {
    return shape.map((d) => {
      if (typeof d === 'string' && d in overrides) return overrides[d];
      return d;
    });
  }
  for (const t of graph.inputs) {
    t.shape = resolveShape(t.shape);
  }
  for (const t of graph.outputs) {
    t.shape = resolveShape(t.shape);
  }
  if (graph.shapes) {
    for (const [name, shape] of graph.shapes) {
      graph.shapes.set(name, resolveShape(shape));
    }
  }
}

/**
 * Replace any remaining symbolic (string) dimensions with a default concrete value.
 * WebNN requires all dimensions to be unsigned long, so dynamic dims must be resolved.
 * Call this after applyFreeDimensionOverrides to handle any dims not covered by overrides.
 */
export function resolveRemainingDynamicDims(
  graph: GraphIR,
  defaultValue = 1,
): void {
  function makeNumeric(shape: (number | string)[]): (number | string)[] {
    return shape.map((d) => (typeof d === 'string' ? defaultValue : d));
  }
  for (const t of graph.inputs) {
    t.shape = makeNumeric(t.shape);
  }
  for (const t of graph.outputs) {
    t.shape = makeNumeric(t.shape);
  }
  if (graph.shapes) {
    for (const [name, shape] of graph.shapes) {
      graph.shapes.set(name, makeNumeric(shape));
    }
  }
}

// Convert tensor/variable name to valid JS identifier
export function toJsVarName(name: string): string {
  let result = name
    .replace(/[^a-zA-Z0-9_]/g, '_')
    .replace(/^(\d)/, 'var_$1');
  if (result === '') result = 'unnamed';
  return result;
}
