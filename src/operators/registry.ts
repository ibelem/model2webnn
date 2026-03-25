// Operator registry — central dispatch from op type → code emitter
// Each op builder is registered here and called during code generation.

import type { NodeIR } from '../ir/graph.js';

// CodeEmitter interface — provided by codegen backends
export interface CodeEmitter {
  // Reference an existing tensor variable name
  ref(tensorName: string): string;
  // Declare a new variable for a tensor
  declare(tensorName: string): string;
  // Emit a line of code
  line(code: string): void;
  // Emit a comment
  comment(text: string): void;
  // Check if a tensor name is a constant (has weights)
  isConstant(tensorName: string): boolean;
  // Get constant info for a tensor
  constantShape(tensorName: string): number[];
  constantDataType(tensorName: string): string;
  // Get raw bytes of a constant tensor (for extracting inline values like padding)
  constantRawData(tensorName: string): Uint8Array | null;
}

export type OpEmitter = (node: NodeIR, emitter: CodeEmitter) => void;

const onnxRegistry = new Map<string, OpEmitter>();
const tfliteRegistry = new Map<string, OpEmitter>();

export function registerOnnxOp(opType: string, emitter: OpEmitter): void {
  onnxRegistry.set(opType, emitter);
}

export function registerOnnxOps(opTypes: string[], emitter: OpEmitter): void {
  for (const op of opTypes) {
    onnxRegistry.set(op, emitter);
  }
}

export function registerTfliteOp(opType: string, emitter: OpEmitter): void {
  tfliteRegistry.set(opType, emitter);
}

export function getOnnxEmitter(opType: string): OpEmitter | undefined {
  return onnxRegistry.get(opType);
}

export function getTfliteEmitter(opType: string): OpEmitter | undefined {
  return tfliteRegistry.get(opType);
}

export function getEmitter(format: string, opType: string): OpEmitter | undefined {
  if (format === 'onnx') return getOnnxEmitter(opType);
  if (format === 'tflite') return getTfliteEmitter(opType);
  return undefined;
}

export function getSupportedOnnxOps(): string[] {
  return [...onnxRegistry.keys()];
}

export function getSupportedTfliteOps(): string[] {
  return [...tfliteRegistry.keys()];
}
