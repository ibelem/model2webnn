// TFLite reduce ops: MEAN, SUM, REDUCE_PROD, REDUCE_MAX, REDUCE_MIN, REDUCE_ANY, REDUCE_ALL

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';
import { emitDequantizeIfNeeded } from './common.js';

function emitReduce(node: NodeIR, emitter: CodeEmitter): void {
  let input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  // WebNN reduce ops require float32/float16 — dequantize if needed
  input = emitDequantizeIfNeeded(input, node.inputs[0], 0, node, emitter, `${output}_in`);

  const opMap: Record<string, string> = {
    MEAN: 'reduceMean',
    SUM: 'reduceSum',
    REDUCE_PROD: 'reduceProduct',
    REDUCE_MAX: 'reduceMax',
    REDUCE_MIN: 'reduceMin',
    REDUCE_ANY: 'reduceSum',   // approximate — no direct WebNN equivalent for bool reduce
    REDUCE_ALL: 'reduceProduct', // approximate
  };

  const webnnOp = opMap[node.opType] ?? 'reduceMean';
  const keepDims = node.attributes.keep_dims as boolean ?? false;

  // Axes come from the second input tensor (a constant int32 tensor)
  // WebNN expects axes as a plain JS array, not an MLOperand
  // Negative axes must be normalized — WebNN axes are unsigned long
  if (node.inputs.length > 1) {
    let axesValues = emitter.constantIntValues(node.inputs[1]);
    if (axesValues) {
      const inputShape = emitter.tensorShape(node.inputs[0]);
      if (inputShape) {
        const rank = inputShape.length;
        axesValues = axesValues.map(a => a < 0 ? a + rank : a);
      }
      emitter.line(`const ${output} = builder.${webnnOp}(${input}, { axes: [${axesValues}], keepDimensions: ${keepDims} });`);
    } else {
      // Fallback: reduce over all axes
      emitter.line(`const ${output} = builder.${webnnOp}(${input}, { keepDimensions: ${keepDims} });`);
    }
  } else {
    emitter.line(`const ${output} = builder.${webnnOp}(${input}, { keepDimensions: ${keepDims} });`);
  }
}

registerTfliteOp('MEAN', emitReduce);
registerTfliteOp('SUM', emitReduce);
registerTfliteOp('REDUCE_PROD', emitReduce);
registerTfliteOp('REDUCE_MAX', emitReduce);
registerTfliteOp('REDUCE_MIN', emitReduce);
registerTfliteOp('REDUCE_ANY', emitReduce);
registerTfliteOp('REDUCE_ALL', emitReduce);
