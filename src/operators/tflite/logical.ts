// TFLite logical / comparison ops: EQUAL, NOT_EQUAL, GREATER, LESS, LOGICAL_AND, etc.

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';

function emitLogicalOp(node: NodeIR, emitter: CodeEmitter): void {
  const output = emitter.declare(node.outputs[0]);

  const binaryOps: Record<string, string> = {
    EQUAL: 'equal',
    NOT_EQUAL: 'equal',  // negate after
    GREATER: 'greater',
    GREATER_EQUAL: 'greaterOrEqual',
    LESS: 'lesser',
    LESS_EQUAL: 'lesserOrEqual',
    LOGICAL_AND: 'logicalAnd',
    LOGICAL_OR: 'logicalOr',
  };

  if (node.opType === 'LOGICAL_NOT') {
    const input = emitter.ref(node.inputs[0]);
    emitter.line(`const ${output} = builder.logicalNot(${input});`);
    return;
  }

  const a = emitter.ref(node.inputs[0]);
  const b = emitter.ref(node.inputs[1]);

  if (node.opType === 'NOT_EQUAL') {
    emitter.line(`const ${output} = builder.logicalNot(builder.equal(${a}, ${b}));`);
  } else {
    const webnnOp = binaryOps[node.opType];
    if (webnnOp) {
      emitter.line(`const ${output} = builder.${webnnOp}(${a}, ${b});`);
    } else {
      emitter.comment(`Unsupported logical op: ${node.opType}`);
      emitter.line(`const ${output} = ${a}; // UNSUPPORTED: ${node.opType}`);
    }
  }
}

registerTfliteOp('EQUAL', emitLogicalOp);
registerTfliteOp('NOT_EQUAL', emitLogicalOp);
registerTfliteOp('GREATER', emitLogicalOp);
registerTfliteOp('GREATER_EQUAL', emitLogicalOp);
registerTfliteOp('LESS', emitLogicalOp);
registerTfliteOp('LESS_EQUAL', emitLogicalOp);
registerTfliteOp('LOGICAL_AND', emitLogicalOp);
registerTfliteOp('LOGICAL_OR', emitLogicalOp);
registerTfliteOp('LOGICAL_NOT', emitLogicalOp);
