// TFLite element-wise binary ops: ADD, SUB, MUL, DIV, POW, MAXIMUM, MINIMUM, FLOOR_DIV

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';
import { emitFusedActivation, emitDequantizeIfNeeded } from './common.js';

function emitBinaryOp(node: NodeIR, emitter: CodeEmitter): void {
  let a = emitter.ref(node.inputs[0]);
  let b = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);

  // WebNN binary ops require float32/float16 — dequantize int8/uint8/int32 inputs
  a = emitDequantizeIfNeeded(a, node.inputs[0], 0, node, emitter, `${output}_a`);
  b = emitDequantizeIfNeeded(b, node.inputs[1], 1, node, emitter, `${output}_b`);

  const opMap: Record<string, string> = {
    ADD: 'add',
    SUB: 'sub',
    MUL: 'mul',
    DIV: 'div',
    POW: 'pow',
    MAXIMUM: 'max',
    MINIMUM: 'min',
    FLOOR_DIV: 'div',  // floor(a / b)
  };

  const webnnOp = opMap[node.opType];
  if (!webnnOp) {
    emitter.comment(`Unsupported binary op: ${node.opType}`);
    emitter.line(`const ${output} = ${a}; // UNSUPPORTED: ${node.opType}`);
    return;
  }

  emitter.comment(node.opType);

  if (node.opType === 'FLOOR_DIV') {
    emitter.line(`const ${output} = builder.floor(builder.div(${a}, ${b}));`);
  } else {
    const rawVar = `${output}_raw`;
    emitter.line(`const ${rawVar} = builder.${webnnOp}(${a}, ${b});`);

    const activation = node.attributes.fused_activation as string | undefined;
    const resultVar = emitFusedActivation(rawVar, activation, emitter);
    if (resultVar !== output) {
      emitter.line(`const ${output} = ${resultVar};`);
    }
  }
}

registerTfliteOp('ADD', emitBinaryOp);
registerTfliteOp('SUB', emitBinaryOp);
registerTfliteOp('MUL', emitBinaryOp);
registerTfliteOp('DIV', emitBinaryOp);
registerTfliteOp('POW', emitBinaryOp);
registerTfliteOp('MAXIMUM', emitBinaryOp);
registerTfliteOp('MINIMUM', emitBinaryOp);
registerTfliteOp('FLOOR_DIV', emitBinaryOp);
