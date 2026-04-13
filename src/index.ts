// Public library API — main entry point for model2webnn
// Converts .onnx / .tflite models into WebNN JavaScript/TypeScript code + WGWT weights.

import type { GraphIR, TensorInfo } from './ir/graph.js';
import { applyFreeDimensionOverrides, getFreeDimensions, resolveRemainingDynamicDims } from './ir/graph.js';
import { foldConstants } from './ir/constant-folding.js';
import { parseOnnx, repropagateReshapeShapes, type ExternalDataMap } from './parsers/onnx.js';
import { parseTflite } from './parsers/tflite.js';
import { packWeights } from './weights/packer.js';
import { generateJavaScriptFixed } from './codegen/javascript.js';
import { generateTypeScript } from './codegen/typescript.js';
import { generateHtml } from './codegen/html.js';
import { generateWebnnDsl } from './codegen/webnn-dsl.js';
import { getEmitter } from './operators/registry.js';
import type { WeightsManifest } from './weights/packer.js';

export type OutputFormat = 'javascript' | 'typescript' | 'html';

/**
 * Minimal model metadata returned by the lightweight parseOnnxMetadata / parseTfliteMetadata
 * functions. Contains only graph input and output tensor descriptions; no weight data.
 */
export interface ModelMetadata {
  inputs: TensorInfo[];
  outputs: TensorInfo[];
}

export interface ConvertOptions {
  format?: OutputFormat;
  weightsFileName?: string;
  manifestFileName?: string;
  modelName?: string;
  /**
   * Optional title override for the generated HTML page.
   * Defaults to `WebNN · <modelName>`.
   * Use this to embed the HuggingFace model ID in the title when the model
   * was fetched from huggingface.co or hf-mirror.com.
   */
  title?: string;
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
  /** WebNN DSL text (.webnn format) */
  webnnDsl: string;
  graph: GraphIR;
  /** Operator coverage analysis — which ops are supported vs unsupported */
  coverage: OperatorCoverage;
  /** Symbolic dimension names that were not overridden and defaulted to 1 */
  unresolvedFreeDims: string[];
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
    title,
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

  // Capture unresolved free dims before they get defaulted to 1
  const unresolvedFreeDims = getFreeDimensions(graph);

  // WebNN requires all dimensions to be concrete unsigned long values.
  // Default any remaining symbolic dimensions to 1.
  resolveRemainingDynamicDims(graph);

  // Re-run shape propagation now that ALL dimensions are concrete (whether from
  // explicit overrides or defaulted to 1). This resolves Reshape outputs whose
  // target shapes depend on formerly-dynamic dims (e.g. Shape → Slice → Concat chains).
  if (modelFormat === 'onnx') {
    repropagateReshapeShapes(graph);
  }

  // Constant folding: evaluate nodes whose inputs are all constants at build time.
  // This resolves shape-computation chains (Shape → Gather → Concat → Reshape)
  // and ops without WebNN equivalents (Range, ConstantOfShape) in transformer models.
  foldConstants(graph);

  // Reinterpret MatMulNBits packed uint8 constants as uint4 with doubled shape.
  // ORT does this in matMulNBits_op_builder.cc — the raw bytes stay the same, but
  // each byte is two uint4 values, so the logical element count doubles.
  fixMatMulNBitsConstants(graph);

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
      html = generateHtml(graph, { weightsFileName, manifestFileName, title });
      break;
    case 'javascript':
    default:
      code = generateJavaScriptFixed(graph, codegenOpts);
      html = generateHtml(graph, { weightsFileName, manifestFileName, title });
      break;
  }

  // Generate WebNN DSL
  const webnnDsl = generateWebnnDsl(graph);

  // Validate operator coverage
  const coverage = validateOperatorCoverage(graph);

  return {
    code,
    weights: packed.weights,
    manifest: packed.manifest,
    html,
    webnnDsl,
    graph,
    coverage,
    unresolvedFreeDims,
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

  // TFLite: FlatBuffers format with "TFL3" file identifier at bytes 4-7
  // Check this FIRST since the first byte (root table offset) can collide
  // with ONNX protobuf field tags
  const fileId = String.fromCharCode(buffer[4], buffer[5], buffer[6], buffer[7]);
  if (fileId === 'TFL3') {
    return 'tflite';
  }

  // ONNX: protobuf format, starts with field 7 (graph) or field 1 (ir_version)
  // Check for protobuf wire format: field number 7, wire type 2 (length-delimited) = 0x3A
  // or field 1, wire type 0 (varint) = 0x08
  if (buffer[0] === 0x08 || buffer[0] === 0x3a || buffer[0] === 0x12) {
    return 'onnx';
  }

  // Fallback: try ONNX (protobuf can start with various field tags)
  return 'onnx';
}

// Re-export types and utilities
export type { GraphIR, TensorInfo, ConstantInfo, NodeIR, MLOperandDataType } from './ir/graph.js';
export { getFreeDimensions, applyFreeDimensionOverrides, resolveRemainingDynamicDims } from './ir/graph.js';
export type { WeightsManifest, PackedWeights } from './weights/packer.js';
export type { ExternalDataMap } from './parsers/onnx.js';
export { parseOnnx, parseOnnxMetadata, getExternalDataRefs } from './parsers/onnx.js';
export { parseTflite, parseTfliteMetadata } from './parsers/tflite.js';
export { packWeights } from './weights/packer.js';
export { generateJavaScriptFixed as generateJavaScript } from './codegen/javascript.js';
export { generateTypeScript } from './codegen/typescript.js';
export { generateHtml } from './codegen/html.js';
// Note: getSupportedOnnxOps and getSupportedTfliteOps are imported at the top and used internally
// Re-export them for downstream consumers:
export { getSupportedOnnxOps, getSupportedTfliteOps } from './operators/registry.js';

/**
 * Reinterpret MatMulNBits packed uint8 constants (B and zero_points) as uint4.
 * Ported from ORT matMulNBits_op_builder.cc: the raw bytes stay the same but
 * WebNN's uint4 type interprets each byte as two 4-bit elements.
 *
 * B:  uint8 [N, n_blocks, blob_size]  → uint4 [N, n_blocks, blob_size * 2]
 * zp: uint8 [N, ceil(n_blocks/2)]     → uint4 [N, n_blocks, 1]
 */
function fixMatMulNBitsConstants(graph: GraphIR): void {
  const constantMap = new Map(graph.constants.map((c) => [c.name, c]));
  const fixed = new Set<string>();

  for (const node of graph.nodes) {
    if (node.opType !== 'MatMulNBits') continue;

    const bName = node.inputs[1];
    const bConst = constantMap.get(bName);
    if (bConst && bConst.dataType === 'uint8' && !fixed.has(bName)) {
      // Double the last dimension: each uint8 byte holds 2 uint4 values
      const lastDim = bConst.shape[bConst.shape.length - 1];
      if (typeof lastDim === 'number') {
        bConst.shape[bConst.shape.length - 1] = lastDim * 2;
      }
      bConst.dataType = 'uint4';
      fixed.add(bName);
      // Update shape/dataType maps if present
      if (graph.shapes) graph.shapes.set(bName, [...bConst.shape]);
      if (graph.dataTypes) graph.dataTypes.set(bName, 'uint4');
    }

    // Zero points (input index 3) — also packed uint8 → uint4
    if (node.inputs.length > 3 && node.inputs[3] !== '') {
      const zpName = node.inputs[3];
      const zpConst = constantMap.get(zpName);
      if (zpConst && zpConst.dataType === 'uint8' && !fixed.has(zpName)) {
        const N = (node.attributes.N as number) ?? 0;
        const K = (node.attributes.K as number) ?? 0;
        const blockSize = (node.attributes.block_size as number) ?? 32;
        const nBlocks = Math.ceil(K / blockSize);
        zpConst.dataType = 'uint4';
        zpConst.shape = [N, nBlocks, 1];
        fixed.add(zpName);
        if (graph.shapes) graph.shapes.set(zpName, [N, nBlocks, 1]);
        if (graph.dataTypes) graph.dataTypes.set(zpName, 'uint4');
      }
    }
  }
}
