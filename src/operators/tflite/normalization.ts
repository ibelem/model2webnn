// TFLite normalization ops: SOFTMAX, LOG_SOFTMAX, L2_NORMALIZATION, LOCAL_RESPONSE_NORMALIZATION

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';
import { emitDequantizeIfNeeded } from './common.js';

function emitSoftmax(node: NodeIR, emitter: CodeEmitter): void {
  let input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  input = emitDequantizeIfNeeded(input, node.inputs[0], 0, node, emitter, `${output}_in`);
  // TFLite softmax always operates on last axis
  emitter.line(`const ${output} = builder.softmax(${input});`);
}

function emitLogSoftmax(node: NodeIR, emitter: CodeEmitter): void {
  let input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  input = emitDequantizeIfNeeded(input, node.inputs[0], 0, node, emitter, `${output}_in`);
  emitter.line(`const ${output} = builder.log(builder.softmax(${input}));`);
}

function emitL2Normalization(node: NodeIR, emitter: CodeEmitter): void {
  let input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  input = emitDequantizeIfNeeded(input, node.inputs[0], 0, node, emitter, `${output}_in`);
  emitter.comment('L2_NORMALIZATION');
  // L2 norm: x / sqrt(sum(x^2))
  const squared = `${output}_sq`;
  const sumSq = `${output}_sum`;
  const sqrtSumSq = `${output}_sqrt`;
  emitter.line(`const ${squared} = builder.mul(${input}, ${input});`);
  emitter.line(`const ${sumSq} = builder.reduceSum(${squared}, { keepDimensions: true });`);
  emitter.line(`const ${sqrtSumSq} = builder.sqrt(${sumSq});`);
  emitter.line(`const ${output} = builder.div(${input}, ${sqrtSumSq});`);
}

function emitLocalResponseNormalization(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.comment('LOCAL_RESPONSE_NORMALIZATION');
  emitter.line(`const ${output} = ${input}; // LRN — requires custom decomposition`);
}

registerTfliteOp('SOFTMAX', emitSoftmax);
registerTfliteOp('LOG_SOFTMAX', emitLogSoftmax);
registerTfliteOp('L2_NORMALIZATION', emitL2Normalization);
registerTfliteOp('LOCAL_RESPONSE_NORMALIZATION', emitLocalResponseNormalization);
