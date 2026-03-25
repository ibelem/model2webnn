// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/binary_op_builder.cc
// 1:1 WebNN mapping for binary operations.

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOps } from '../registry.js';

function emitBinary(node: NodeIR, emitter: CodeEmitter): void {
  const lhs = emitter.ref(node.inputs[0]);
  const rhs = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);

  const opMap: Record<string, string> = {
    Add: 'add',
    Sub: 'sub',
    Mul: 'mul',
    Div: 'div',
    Pow: 'pow',
    PRelu: 'prelu',
  };

  const webnnOp = opMap[node.opType];
  if (!webnnOp) {
    emitter.comment(`TODO: Unsupported binary op: ${node.opType}`);
    return;
  }

  emitter.line(`const ${output} = builder.${webnnOp}(${lhs}, ${rhs});`);
}

registerOnnxOps(['Add', 'Sub', 'Mul', 'Div', 'Pow', 'PRelu'], emitBinary);
