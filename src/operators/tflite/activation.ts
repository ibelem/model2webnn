// TFLite activation ops: RELU, RELU6, LOGISTIC, TANH, ELU, HARD_SWISH, LEAKY_RELU, GELU, PRELU, etc.

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';
import { emitDequantizeIfNeeded } from './common.js';

function emitSimpleActivation(node: NodeIR, emitter: CodeEmitter): void {
  let input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  // WebNN activation ops require float32/float16 — dequantize int8/uint8/int32 inputs
  input = emitDequantizeIfNeeded(input, node.inputs[0], 0, node, emitter, `${output}_in`);

  switch (node.opType) {
    case 'RELU':
      emitter.line(`const ${output} = builder.relu(${input});`);
      break;
    case 'RELU6':
      emitter.line(`const ${output} = builder.clamp(${input}, { minValue: 0, maxValue: 6 });`);
      break;
    case 'RELU_N1_TO_1':
      emitter.line(`const ${output} = builder.clamp(${input}, { minValue: -1, maxValue: 1 });`);
      break;
    case 'RELU_0_TO_1':
      emitter.line(`const ${output} = builder.clamp(${input}, { minValue: 0, maxValue: 1 });`);
      break;
    case 'LOGISTIC':
      emitter.line(`const ${output} = builder.sigmoid(${input});`);
      break;
    case 'TANH':
      emitter.line(`const ${output} = builder.tanh(${input});`);
      break;
    case 'ELU':
      emitter.line(`const ${output} = builder.elu(${input});`);
      break;
    case 'HARD_SWISH':
      emitter.line(`const ${output} = builder.hardSwish(${input});`);
      break;
    case 'LEAKY_RELU': {
      const alpha = (node.attributes.alpha as number) ?? 0.01;
      emitter.line(`const ${output} = builder.leakyRelu(${input}, { alpha: ${alpha} });`);
      break;
    }
    case 'GELU':
      emitter.line(`const ${output} = builder.gelu(${input});`);
      break;
    case 'PRELU': {
      let slope = emitter.ref(node.inputs[1]);
      slope = emitDequantizeIfNeeded(slope, node.inputs[1], 1, node, emitter, `${output}_slope`);
      emitter.line(`const ${output} = builder.prelu(${input}, ${slope});`);
      break;
    }
    default:
      emitter.comment(`Unsupported activation: ${node.opType}`);
      emitter.line(`const ${output} = ${input}; // UNSUPPORTED: ${node.opType}`);
  }
}

registerTfliteOp('RELU', emitSimpleActivation);
registerTfliteOp('RELU6', emitSimpleActivation);
registerTfliteOp('RELU_N1_TO_1', emitSimpleActivation);
registerTfliteOp('RELU_0_TO_1', emitSimpleActivation);
registerTfliteOp('LOGISTIC', emitSimpleActivation);
registerTfliteOp('TANH', emitSimpleActivation);
registerTfliteOp('ELU', emitSimpleActivation);
registerTfliteOp('HARD_SWISH', emitSimpleActivation);
registerTfliteOp('LEAKY_RELU', emitSimpleActivation);
registerTfliteOp('GELU', emitSimpleActivation);
registerTfliteOp('PRELU', emitSimpleActivation);
