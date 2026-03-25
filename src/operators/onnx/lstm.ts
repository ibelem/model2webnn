// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/lstm_op_builder.cc
// ONNX LSTM → WebNN builder.lstm()

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

function emitLSTM(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);       // [steps, batch, input_size]
  const weight = emitter.ref(node.inputs[1]);       // [num_dir, 4*hidden, input_size]
  const recurrentWeight = emitter.ref(node.inputs[2]); // [num_dir, 4*hidden, hidden_size]
  const output = node.outputs[0] !== '' ? emitter.declare(node.outputs[0]) : null;

  const hiddenSize = (node.attributes.hidden_size as number) ?? 1;
  const direction = (node.attributes.direction as string) ?? 'forward';

  emitter.comment(`LSTM — hidden_size=${hiddenSize}, direction=${direction}`);

  // Map ONNX direction to WebNN
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

  // Layout (ORT uses 'iofg')
  opts.push(`layout: 'iofg'`);

  // Return sequence if output Y exists
  const returnSequence = node.outputs.length > 2 && node.outputs[2] !== '';
  opts.push(`returnSequence: ${returnSequence}`);

  // Bias — split ONNX bias [num_dir, 8*hidden] into [bias, recurrentBias]
  const hasBias = node.inputs.length > 3 && node.inputs[3] !== '';
  if (hasBias) {
    const bias = emitter.ref(node.inputs[3]);
    const biasSplit = `${(output ?? 'lstm')}_bias_split`;
    emitter.line(`const ${biasSplit} = builder.split(${bias}, 2, { axis: 1 });`);
    opts.push(`bias: ${biasSplit}[0]`);
    opts.push(`recurrentBias: ${biasSplit}[1]`);
  }

  // Initial hidden state
  if (node.inputs.length > 5 && node.inputs[5] !== '') {
    opts.push(`initialHiddenState: ${emitter.ref(node.inputs[5])}`);
  }

  // Initial cell state
  if (node.inputs.length > 6 && node.inputs[6] !== '') {
    opts.push(`initialCellState: ${emitter.ref(node.inputs[6])}`);
  }

  // Peephole weights
  if (node.inputs.length > 7 && node.inputs[7] !== '') {
    opts.push(`peepholeWeight: ${emitter.ref(node.inputs[7])}`);
  }

  // steps (first dim of input — we use dynamic value)
  const stepsVar = `${(output ?? 'lstm')}_steps`;
  emitter.line(`const ${stepsVar} = ${input}.shape[0];`);

  const resultVar = `${(output ?? 'lstm')}_result`;
  emitter.line(`const ${resultVar} = builder.lstm(${input}, ${weight}, ${recurrentWeight}, ${stepsVar}, ${hiddenSize}, { ${opts.join(', ')} });`);

  // Map outputs: [Y_h, Y_c, Y] from WebNN → [Y, Y_h, Y_c] from ONNX
  if (node.outputs.length > 0 && node.outputs[0] !== '') {
    const yh = emitter.declare(node.outputs[0]);
    emitter.line(`const ${yh} = ${resultVar}[0];`);
  }
  if (node.outputs.length > 1 && node.outputs[1] !== '') {
    const yc = emitter.declare(node.outputs[1]);
    emitter.line(`const ${yc} = ${resultVar}[1];`);
  }
  if (node.outputs.length > 2 && node.outputs[2] !== '') {
    const y = emitter.declare(node.outputs[2]);
    emitter.line(`const ${y} = ${resultVar}[2];`);
  }
}

registerOnnxOp('LSTM', emitLSTM);
