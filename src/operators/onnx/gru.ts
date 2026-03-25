// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/gru_op_builder.cc
// ONNX GRU → WebNN builder.gru()

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

function emitGRU(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);       // [steps, batch, input_size]
  const weight = emitter.ref(node.inputs[1]);       // [num_dir, 3*hidden, input_size]
  const recurrentWeight = emitter.ref(node.inputs[2]); // [num_dir, 3*hidden, hidden_size]
  const output = node.outputs[0] !== '' ? emitter.declare(node.outputs[0]) : null;

  const hiddenSize = (node.attributes.hidden_size as number) ?? 1;
  const direction = (node.attributes.direction as string) ?? 'forward';

  emitter.comment(`GRU — hidden_size=${hiddenSize}, direction=${direction}`);

  const webnnDirection = direction === 'bidirectional' ? 'both'
    : direction === 'reverse' ? 'backward' : 'forward';

  const opts: string[] = [];
  opts.push(`direction: '${webnnDirection}'`);

  // Activations
  const activations = node.attributes.activations as string[] | undefined;
  if (activations?.length) {
    const mapped = activations.map((a) => `'${a.toLowerCase()}'`);
    opts.push(`activations: [${mapped.join(', ')}]`);
  }

  // Layout (zrn = update, reset, new)
  opts.push(`layout: 'zrn'`);

  // linear_before_reset → resetAfter
  const linearBeforeReset = (node.attributes.linear_before_reset as number) ?? 0;
  opts.push(`resetAfter: ${!!linearBeforeReset}`);

  // Return sequence if output Y exists
  const returnSequence = node.outputs.length > 1 && node.outputs[1] !== '';
  opts.push(`returnSequence: ${returnSequence}`);

  // Bias — split ONNX bias [num_dir, 6*hidden] into [bias, recurrentBias]
  const hasBias = node.inputs.length > 3 && node.inputs[3] !== '';
  if (hasBias) {
    const bias = emitter.ref(node.inputs[3]);
    const biasSplit = `${(output ?? 'gru')}_bias_split`;
    emitter.line(`const ${biasSplit} = builder.split(${bias}, 2, { axis: 1 });`);
    opts.push(`bias: ${biasSplit}[0]`);
    opts.push(`recurrentBias: ${biasSplit}[1]`);
  }

  // Initial hidden state
  if (node.inputs.length > 5 && node.inputs[5] !== '') {
    opts.push(`initialHiddenState: ${emitter.ref(node.inputs[5])}`);
  }

  const stepsVar = `${(output ?? 'gru')}_steps`;
  emitter.line(`const ${stepsVar} = ${input}.shape[0];`);

  const resultVar = `${(output ?? 'gru')}_result`;
  emitter.line(`const ${resultVar} = builder.gru(${input}, ${weight}, ${recurrentWeight}, ${stepsVar}, ${hiddenSize}, { ${opts.join(', ')} });`);

  // Outputs: [Y_h, Y]
  if (node.outputs.length > 0 && node.outputs[0] !== '') {
    const yh = emitter.declare(node.outputs[0]);
    emitter.line(`const ${yh} = ${resultVar}[0];`);
  }
  if (node.outputs.length > 1 && node.outputs[1] !== '') {
    const y = emitter.declare(node.outputs[1]);
    emitter.line(`const ${y} = ${resultVar}[1];`);
  }
}

registerOnnxOp('GRU', emitGRU);
