// Public library API — main entry point for model2webnn
// Converts .onnx / .tflite models into WebNN JavaScript/TypeScript code + WGWT weights.

import type { GraphIR } from './ir/graph.js';
import { applyFreeDimensionOverrides } from './ir/graph.js';
import { parseOnnx, type ExternalDataMap } from './parsers/onnx.js';
import { parseTflite } from './parsers/tflite.js';
import { packWeights } from './weights/packer.js';
import { generateJavaScriptFixed } from './codegen/javascript.js';
import { generateTypeScript } from './codegen/typescript.js';
import { generateHtml } from './codegen/html.js';
import { getEmitter } from './operators/registry.js';
import type { WeightsManifest } from './weights/packer.js';

export type OutputFormat = 'javascript' | 'typescript' | 'html';

export interface ConvertOptions {
  format?: OutputFormat;
  weightsFileName?: string;
  manifestFileName?: string;
  modelName?: string;
  /** External data files for ONNX models with weights in separate files */
  externalData?: ExternalDataMap;
  /**
   * Override symbolic (dynamic) dimensions with fixed values.
   * Keys are dimension parameter names (e.g. "batch_size", "sequence_length"),
   * values are the fixed integer sizes to use.
   * See: https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#freedimensionoverrides
   */
  freeDimensionOverrides?: Record<string, number>;
}

export interface OperatorCoverage {
  /** Total number of ops used in the model */
  totalOps: number;
  /** Number of supported ops */
  supportedOps: number;
  /** Number of unsupported ops */
  unsupportedOps: number;
  /** Coverage percentage (0-100) */
  coveragePercent: number;
  /** Unsupported op types with occurrence counts */
  unsupportedOpTypes: { opType: string; count: number }[];
  /** All op types used in the model with their support status */
  opDetails: { opType: string; count: number; supported: boolean }[];
}

export interface ConvertResult {
  code: string;
  weights: Uint8Array;
  manifest: WeightsManifest;
  html?: string;
  graph: GraphIR;
  /** Operator coverage analysis — which ops are supported vs unsupported */
  coverage: OperatorCoverage;
}

/**
 * Convert a model buffer to WebNN code + weights.
 *
 * @param buffer - Raw model file bytes (.onnx or .tflite)
 * @param options - Output options
 * @returns Generated code, weights binary, manifest, and optionally HTML
 */
export async function convert(
  buffer: Uint8Array,
  options: ConvertOptions = {},
): Promise<ConvertResult> {
  const {
    format = 'javascript',
    weightsFileName = 'model.weights',
    manifestFileName = 'model.manifest.json',
    modelName,
    externalData,
    freeDimensionOverrides,
  } = options;

  // Detect model format from magic bytes
  const modelFormat = detectFormat(buffer);

  // Parse model to IR
  let graph: GraphIR;
  if (modelFormat === 'onnx') {
    graph = await parseOnnx(buffer, externalData);
  } else if (modelFormat === 'tflite') {
    graph = await parseTflite(buffer);
  } else {
    throw new Error(`Unknown model format. Expected .onnx or .tflite file.`);
  }

  if (modelName) {
    graph.name = modelName;
  }

  // Apply free dimension overrides if provided
  if (freeDimensionOverrides && Object.keys(freeDimensionOverrides).length > 0) {
    applyFreeDimensionOverrides(graph, freeDimensionOverrides);
  }

  // Pack weights into WGWT format
  const packed = packWeights(graph.constants);

  // Generate code
  const codegenOpts = { weightsFileName, manifestFileName };
  let code: string;
  let html: string | undefined;

  switch (format) {
    case 'typescript':
      code = generateTypeScript(graph, codegenOpts);
      break;
    case 'html':
      code = generateJavaScriptFixed(graph, codegenOpts);
      html = generateHtml(graph, { weightsFileName, manifestFileName });
      break;
    case 'javascript':
    default:
      code = generateJavaScriptFixed(graph, codegenOpts);
      html = generateHtml(graph, { weightsFileName, manifestFileName });
      break;
  }

  // Validate operator coverage
  const coverage = validateOperatorCoverage(graph);

  return {
    code,
    weights: packed.weights,
    manifest: packed.manifest,
    html,
    graph,
    coverage,
  };
}

/**
 * Validate operator coverage for a parsed graph.
 * Returns detailed information about which ops are supported and which are not.
 */
export function validateOperatorCoverage(graph: GraphIR): OperatorCoverage {
  const opCounts = new Map<string, number>();
  for (const node of graph.nodes) {
    opCounts.set(node.opType, (opCounts.get(node.opType) ?? 0) + 1);
  }

  const opDetails: OperatorCoverage['opDetails'] = [];
  const unsupportedOpTypes: OperatorCoverage['unsupportedOpTypes'] = [];
  let supportedCount = 0;
  let unsupportedCount = 0;

  for (const [opType, count] of opCounts) {
    const supported = !!getEmitter(graph.format, opType);
    opDetails.push({ opType, count, supported });
    if (supported) {
      supportedCount += count;
    } else {
      unsupportedCount += count;
      unsupportedOpTypes.push({ opType, count });
    }
  }

  const totalOps = graph.nodes.length;
  return {
    totalOps,
    supportedOps: supportedCount,
    unsupportedOps: unsupportedCount,
    coveragePercent: totalOps > 0 ? Math.round((supportedCount / totalOps) * 100) : 100,
    unsupportedOpTypes: unsupportedOpTypes.sort((a, b) => b.count - a.count),
    opDetails: opDetails.sort((a, b) => b.count - a.count),
  };
}

/**
 * Detect model format from file magic bytes.
 */
export function detectFormat(buffer: Uint8Array): 'onnx' | 'tflite' | 'unknown' {
  if (buffer.length < 8) return 'unknown';

  // ONNX: protobuf format, starts with field 7 (graph) or field 1 (ir_version)
  // Check for protobuf wire format: field number 7, wire type 2 (length-delimited) = 0x3A
  // or field 1, wire type 0 (varint) = 0x08
  if (buffer[0] === 0x08 || buffer[0] === 0x3a || buffer[0] === 0x12) {
    return 'onnx';
  }

  // TFLite: FlatBuffers format — no universal magic, but typically starts with
  // a root table offset. We check for common patterns.
  // FlatBuffers files usually have the file identifier at bytes 4-7
  // TFLite uses "TFL3" as file identifier
  const fileId = String.fromCharCode(buffer[4], buffer[5], buffer[6], buffer[7]);
  if (fileId === 'TFL3') {
    return 'tflite';
  }

  // Fallback: try ONNX (protobuf can start with various field tags)
  return 'onnx';
}

// Re-export types and utilities
export type { GraphIR, TensorInfo, ConstantInfo, NodeIR, MLOperandDataType } from './ir/graph.js';
export { getFreeDimensions, applyFreeDimensionOverrides } from './ir/graph.js';
export type { WeightsManifest, PackedWeights } from './weights/packer.js';
export type { ExternalDataMap } from './parsers/onnx.js';
export { parseOnnx, getExternalDataRefs } from './parsers/onnx.js';
export { parseTflite } from './parsers/tflite.js';
export { packWeights } from './weights/packer.js';
export { generateJavaScriptFixed as generateJavaScript } from './codegen/javascript.js';
export { generateTypeScript } from './codegen/typescript.js';
export { generateHtml } from './codegen/html.js';
// Note: getSupportedOnnxOps and getSupportedTfliteOps are imported at the top and used internally
// Re-export them for downstream consumers:
export { getSupportedOnnxOps, getSupportedTfliteOps } from './operators/registry.js';
