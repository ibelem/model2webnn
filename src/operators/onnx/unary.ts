// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/unary_op_builder.cc
// 1:1 WebNN mapping for unary operations.

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOps } from '../registry.js';

function emitUnary(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const opMap: Record<string, string> = {
    Abs: 'abs',
    Ceil: 'ceil',
    Cos: 'cos',
    Erf: 'erf',
    Exp: 'exp',
    Floor: 'floor',
    Identity: 'identity',
    Log: 'log',
    Neg: 'neg',
    Reciprocal: 'reciprocal',
    Round: 'round',
    Sign: 'sign',
    Sin: 'sin',
    Sqrt: 'sqrt',
    Tan: 'tan',
  };

  const webnnOp = opMap[node.opType];
  if (!webnnOp) {
    emitter.comment(`TODO: Unsupported unary op: ${node.opType}`);
    return;
  }

  if (node.opType === 'Identity') {
    // Identity is a no-op — just alias the variable
    emitter.line(`const ${output} = ${input};`);
  } else {
    emitter.line(`const ${output} = builder.${webnnOp}(${input});`);
  }
}

registerOnnxOps(
  [
    'Abs', 'Ceil', 'Cos', 'Erf', 'Exp', 'Floor', 'Identity',
    'Log', 'Neg', 'Reciprocal', 'Round', 'Sign', 'Sin', 'Sqrt', 'Tan',
  ],
  emitUnary,
);
