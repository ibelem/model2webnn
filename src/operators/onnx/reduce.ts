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

  // Axes from attribute (opset <18) or second input constant (opset 18+ / ReduceSum opset 13+)
  let axes = node.attributes.axes as number[] | undefined;
  const inputShape = emitter.tensorShape(node.inputs[0]);
  const rank = inputShape ? inputShape.length : 0;
  if (axes && axes.some((a) => a < 0)) {
    // Resolve negative attribute axes using input rank
    if (rank > 0) {
      axes = axes.map((a) => (a < 0 ? a + rank : a));
    }
  }
  if (!axes && node.inputs.length > 1 && node.inputs[1] !== '' && emitter.isConstant(node.inputs[1])) {
    const axesValues = emitter.constantIntValues(node.inputs[1]);
    if (axesValues && axesValues.length > 0) {
      axes = axesValues.map((a) => (a < 0 && rank > 0 ? a + rank : a));
    }
  }

  const noopWithEmptyAxes = (node.attributes.noop_with_empty_axes as number) ?? 0;

  const opts: string[] = [];
  if (axes && axes.length > 0) {
    opts.push(`axes: [${axes.join(', ')}]`);
  } else if (noopWithEmptyAxes) {
    // noop_with_empty_axes=1: pass empty axes array (no reduction, but other ops still run)
    opts.push(`axes: []`);
  }
  if (keepdims) {
    opts.push(`keepDimensions: true`);
  }

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.${webnnOp}(${input}${optsStr});`);
}

for (const op of Object.keys(onnxToWebnn)) {
  registerOnnxOp(op, emitReduce);
}
