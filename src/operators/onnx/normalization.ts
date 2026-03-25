// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/normalization_op_builder.cc
// Maps: BatchNormalization, LayerNormalization, InstanceNormalization, GroupNormalization

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

// BatchNormalization → batchNormalization
function emitBatchNormalization(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const scale = emitter.ref(node.inputs[1]);
  const bias = emitter.ref(node.inputs[2]);
  const mean = emitter.ref(node.inputs[3]);
  const variance = emitter.ref(node.inputs[4]);
  const output = emitter.declare(node.outputs[0]);

  const epsilon = (node.attributes.epsilon as number) ?? 1e-5;
  const opts: string[] = [`scale: ${scale}`, `bias: ${bias}`];
  if (epsilon !== 1e-5) {
    opts.push(`epsilon: ${epsilon}`);
  }

  emitter.line(`const ${output} = builder.batchNormalization(${input}, ${mean}, ${variance}, { ${opts.join(', ')} });`);
}

// LayerNormalization → layerNormalization
function emitLayerNormalization(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const axis = (node.attributes.axis as number) ?? -1;
  const epsilon = (node.attributes.epsilon as number) ?? 1e-5;
  const opts: string[] = [];

  if (axis !== -1) opts.push(`axes: [${axis}]`);
  if (epsilon !== 1e-5) opts.push(`epsilon: ${epsilon}`);

  const hasScale = node.inputs.length > 1 && node.inputs[1] !== '';
  const hasBias = node.inputs.length > 2 && node.inputs[2] !== '';
  if (hasScale) opts.push(`scale: ${emitter.ref(node.inputs[1])}`);
  if (hasBias) opts.push(`bias: ${emitter.ref(node.inputs[2])}`);

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.layerNormalization(${input}${optsStr});`);
}

// InstanceNormalization → instanceNormalization
function emitInstanceNormalization(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const scale = emitter.ref(node.inputs[1]);
  const bias = emitter.ref(node.inputs[2]);
  const output = emitter.declare(node.outputs[0]);

  const epsilon = (node.attributes.epsilon as number) ?? 1e-5;
  const opts: string[] = [`scale: ${scale}`, `bias: ${bias}`];
  if (epsilon !== 1e-5) opts.push(`epsilon: ${epsilon}`);

  emitter.line(`const ${output} = builder.instanceNormalization(${input}, { ${opts.join(', ')} });`);
}

// GroupNormalization → layerNormalization (decomposed per ORT)
function emitGroupNormalization(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const scale = emitter.ref(node.inputs[1]);
  const bias = emitter.ref(node.inputs[2]);
  const output = emitter.declare(node.outputs[0]);

  const epsilon = (node.attributes.epsilon as number) ?? 1e-5;
  const numGroups = (node.attributes.num_groups as number) ?? 1;

  emitter.comment(`GroupNormalization: ${numGroups} groups`);
  const opts: string[] = [`scale: ${scale}`, `bias: ${bias}`];
  if (epsilon !== 1e-5) opts.push(`epsilon: ${epsilon}`);
  opts.push(`groups: ${numGroups}`);

  emitter.line(`const ${output} = builder.layerNormalization(${input}, { ${opts.join(', ')} });`);
}

registerOnnxOp('BatchNormalization', emitBatchNormalization);
registerOnnxOp('LayerNormalization', emitLayerNormalization);
registerOnnxOp('InstanceNormalization', emitInstanceNormalization);
registerOnnxOp('GroupNormalization', emitGroupNormalization);

// SimplifiedLayerNormalization — like LayerNormalization but no bias, no mean subtraction
// normalization_op_builder.cc: maps to builder.layerNormalization with no bias
function emitSimplifiedLayerNormalization(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const axis = (node.attributes.axis as number) ?? -1;
  const epsilon = (node.attributes.epsilon as number) ?? 1e-5;
  const opts: string[] = [];

  if (axis !== -1) opts.push(`axes: [${axis}]`);
  if (epsilon !== 1e-5) opts.push(`epsilon: ${epsilon}`);

  const hasScale = node.inputs.length > 1 && node.inputs[1] !== '';
  if (hasScale) opts.push(`scale: ${emitter.ref(node.inputs[1])}`);

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.layerNormalization(${input}${optsStr});`);
}
registerOnnxOp('SimplifiedLayerNormalization', emitSimplifiedLayerNormalization);

// SkipSimplifiedLayerNormalization — skip connection + simplified layer norm
// Decomposes to: t = input + skip; output = layerNorm(t)
function emitSkipSimplifiedLayerNormalization(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const skip = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);

  const epsilon = (node.attributes.epsilon as number) ?? 1e-5;

  // Add skip connection
  const added = `${output}_skip`;
  emitter.line(`const ${added} = builder.add(${input}, ${skip});`);

  // Optional bias (input[2])
  let normalized_input = added;
  if (node.inputs.length > 2 && node.inputs[2] !== '') {
    const bias = emitter.ref(node.inputs[2]);
    const biased = `${output}_biased`;
    emitter.line(`const ${biased} = builder.add(${added}, ${bias});`);
    normalized_input = biased;
  }

  const opts: string[] = [];
  if (epsilon !== 1e-5) opts.push(`epsilon: ${epsilon}`);

  // Scale (input[3])
  if (node.inputs.length > 3 && node.inputs[3] !== '') {
    opts.push(`scale: ${emitter.ref(node.inputs[3])}`);
  }

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.layerNormalization(${normalized_input}${optsStr});`);

  // Second output is the skip+bias result itself (for residual paths)
  if (node.outputs.length > 1 && node.outputs[1] !== '') {
    const skipOut = emitter.declare(node.outputs[1]);
    emitter.line(`const ${skipOut} = ${normalized_input};`);
  }
}
registerOnnxOp('SkipSimplifiedLayerNormalization', emitSkipSimplifiedLayerNormalization);
