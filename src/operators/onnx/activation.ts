// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/activation_op_builder.cc
// 1:1 WebNN mapping for activation operations.

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOps } from '../registry.js';

function emitActivation(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  switch (node.opType) {
    case 'Relu':
      emitter.line(`const ${output} = builder.relu(${input});`);
      break;
    case 'Sigmoid':
      emitter.line(`const ${output} = builder.sigmoid(${input});`);
      break;
    case 'Tanh':
      emitter.line(`const ${output} = builder.tanh(${input});`);
      break;
    case 'Elu': {
      const alpha = (node.attributes.alpha as number) ?? 1.0;
      emitter.line(`const ${output} = builder.elu(${input}, { alpha: ${alpha} });`);
      break;
    }
    case 'Gelu':
      emitter.line(`const ${output} = builder.gelu(${input});`);
      break;
    case 'HardSigmoid': {
      const alpha = (node.attributes.alpha as number) ?? 0.2;
      const beta = (node.attributes.beta as number) ?? 0.5;
      emitter.line(`const ${output} = builder.hardSigmoid(${input}, { alpha: ${alpha}, beta: ${beta} });`);
      break;
    }
    case 'HardSwish':
      emitter.line(`const ${output} = builder.hardSwish(${input});`);
      break;
    case 'LeakyRelu': {
      const alpha = (node.attributes.alpha as number) ?? 0.01;
      emitter.line(`const ${output} = builder.leakyRelu(${input}, { alpha: ${alpha} });`);
      break;
    }
    case 'Softplus':
      emitter.line(`const ${output} = builder.softplus(${input});`);
      break;
    case 'Softsign':
      emitter.line(`const ${output} = builder.softsign(${input});`);
      break;
    default:
      emitter.comment(`TODO: Unsupported activation: ${node.opType}`);
  }
}

registerOnnxOps(
  [
    'Relu', 'Sigmoid', 'Tanh', 'Elu', 'Gelu', 'HardSigmoid',
    'HardSwish', 'LeakyRelu', 'Softplus', 'Softsign',
  ],
  emitActivation,
);
