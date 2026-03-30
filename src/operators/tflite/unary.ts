// TFLite unary math ops: ABS, CEIL, COS, EXP, FLOOR, LOG, NEG, SIN, SQRT, RSQRT, etc.
// Also includes SQUARED_DIFFERENCE (composite unary-like op)

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';
import { emitDequantizeIfNeeded } from './common.js';

function emitUnaryOp(node: NodeIR, emitter: CodeEmitter): void {
  let input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  // WebNN math ops require float32/float16 — dequantize int8/uint8/int32 inputs
  input = emitDequantizeIfNeeded(input, node.inputs[0], 0, node, emitter, `${output}_in`);

  const opMap: Record<string, string> = {
    ABS: 'abs',
    CEIL: 'ceil',
    COS: 'cos',
    EXP: 'exp',
    FLOOR: 'floor',
    LOG: 'log',
    NEG: 'neg',
    SIN: 'sin',
    SQRT: 'sqrt',
    RSQRT: 'reciprocal', // rsqrt(x) = 1/sqrt(x)
    ROUND: 'identity',   // placeholder — WebNN doesn't have round
    SIGN: 'sign',
    SQUARE: 'pow',        // x^2
  };

  if (node.opType === 'RSQRT') {
    emitter.comment('RSQRT → reciprocal(sqrt(x))');
    emitter.line(`const ${output} = builder.reciprocal(builder.sqrt(${input}));`);
  } else if (node.opType === 'SQUARE') {
    emitter.comment('SQUARE → pow(x, 2)');
    const twoConst = `${output}_two`;
    emitter.line(`const ${twoConst} = builder.constant({ dataType: 'float32', shape: [] }, new Float32Array([2]));`);
    emitter.line(`const ${output} = builder.pow(${input}, ${twoConst});`);
  } else {
    const webnnOp = opMap[node.opType];
    if (webnnOp && webnnOp !== 'identity') {
      emitter.line(`const ${output} = builder.${webnnOp}(${input});`);
    } else {
      emitter.line(`const ${output} = ${input}; // ${node.opType} (identity/passthrough)`);
    }
  }
}

function emitSquaredDifference(node: NodeIR, emitter: CodeEmitter): void {
  let a = emitter.ref(node.inputs[0]);
  let b = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);

  // WebNN sub/mul require float — dequantize if needed
  a = emitDequantizeIfNeeded(a, node.inputs[0], 0, node, emitter, `${output}_a`);
  b = emitDequantizeIfNeeded(b, node.inputs[1], 1, node, emitter, `${output}_b`);

  emitter.comment('SQUARED_DIFFERENCE → (a - b) * (a - b)');
  const diff = `${output}_diff`;
  emitter.line(`const ${diff} = builder.sub(${a}, ${b});`);
  emitter.line(`const ${output} = builder.mul(${diff}, ${diff});`);
}

registerTfliteOp('ABS', emitUnaryOp);
registerTfliteOp('CEIL', emitUnaryOp);
registerTfliteOp('COS', emitUnaryOp);
registerTfliteOp('EXP', emitUnaryOp);
registerTfliteOp('FLOOR', emitUnaryOp);
registerTfliteOp('LOG', emitUnaryOp);
registerTfliteOp('NEG', emitUnaryOp);
registerTfliteOp('SIN', emitUnaryOp);
registerTfliteOp('SQRT', emitUnaryOp);
registerTfliteOp('RSQRT', emitUnaryOp);
registerTfliteOp('ROUND', emitUnaryOp);
registerTfliteOp('SIGN', emitUnaryOp);
registerTfliteOp('SQUARE', emitUnaryOp);
registerTfliteOp('SQUARED_DIFFERENCE', emitSquaredDifference);
