// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/reduction_op_builder.cc
// Maps ONNX Reduce* ops → WebNN reduce* equivalents

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

const onnxToWebnn: Record<string, string> = {
  ReduceMax: 'reduceMax',
  ReduceMean: 'reduceMean',
  ReduceMin: 'reduceMin',
  ReduceProd: 'reduceProduct',
  ReduceSum: 'reduceSum',
  ReduceSumSquare: 'reduceSumSquare',
  ReduceL1: 'reduceL1',
  ReduceL2: 'reduceL2',
  ReduceLogSum: 'reduceLogSum',
  ReduceLogSumExp: 'reduceLogSumExp',
};

function emitReduce(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const webnnOp = onnxToWebnn[node.opType] ?? 'reduceMean';

  const keepdims = (node.attributes.keepdims as number) ?? 1;
  const axes = node.attributes.axes as number[] | undefined;

  const opts: string[] = [];
  if (axes) {
    opts.push(`axes: [${axes.join(', ')}]`);
  } else if (node.inputs.length > 1 && node.inputs[1] !== '') {
    // ONNX opset 18+: axes as second input (handled at code level)
  }
  if (!keepdims) {
    opts.push(`keepDimensions: false`);
  }

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.${webnnOp}(${input}${optsStr});`);
}

for (const op of Object.keys(onnxToWebnn)) {
  registerOnnxOp(op, emitReduce);
}
