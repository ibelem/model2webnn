// TFLite dense/matmul ops: FULLY_CONNECTED, BATCH_MATMUL

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';
import { emitFusedActivation, emitDequantizeIfNeeded } from './common.js';

function emitFullyConnected(node: NodeIR, emitter: CodeEmitter): void {
  let input = emitter.ref(node.inputs[0]);
  let weights = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  let bias = node.inputs.length > 2 ? emitter.ref(node.inputs[2]) : undefined;

  emitter.comment('FULLY_CONNECTED → gemm');

  // WebNN gemm only supports float32/float16. If inputs are int8/uint8/int32 (quantized model),
  // dequantize them to float32 first using the tensor's quantization parameters.
  input = emitDequantizeIfNeeded(input, node.inputs[0], 0, node, emitter, `${output}_a`);
  weights = emitDequantizeIfNeeded(weights, node.inputs[1], 1, node, emitter, `${output}_b`);
  if (bias) {
    bias = emitDequantizeIfNeeded(bias, node.inputs[2], 2, node, emitter, `${output}_c`);
  }

  // WebNN gemm requires rank-2 inputs. TFLite FULLY_CONNECTED accepts any rank
  // and implicitly flattens the input to 2D: [batch, features].
  const inputShape = emitter.tensorShape(node.inputs[0]);
  const outputShape = emitter.tensorShape(node.outputs[0]);
  let needsReshapeBack = false;
  if (inputShape && inputShape.length > 2) {
    // Flatten to [batch, rest] — batch is product of all dims except last
    const lastDim = inputShape[inputShape.length - 1];
    if (typeof lastDim === 'number') {
      const batchDim = inputShape.slice(0, -1).reduce((acc: number, d) =>
        typeof d === 'number' ? acc * d : acc, 1);
      const flatVar = `${output}_flat`;
      emitter.line(`const ${flatVar} = builder.reshape(${input}, [${batchDim}, ${lastDim}]);`);
      input = flatVar;
      needsReshapeBack = !!(outputShape && outputShape.length > 2);
    }
  }

  // TFLite FullyConnected does: output = input * weights^T + bias
  const rawVar = `${output}_raw`;
  if (bias) {
    emitter.line(`const ${rawVar} = builder.gemm(${input}, ${weights}, { c: ${bias}, bTranspose: true });`);
  } else {
    emitter.line(`const ${rawVar} = builder.gemm(${input}, ${weights}, { bTranspose: true });`);
  }

  let resultVar = rawVar;
  const activation = node.attributes.fused_activation as string | undefined;
  resultVar = emitFusedActivation(resultVar, activation, emitter);

  // Reshape back to the expected output shape when input was flattened from >2D
  if (needsReshapeBack && outputShape) {
    const reshapeVar = resultVar === output ? `${output}_2d` : resultVar;
    if (reshapeVar !== resultVar) {
      emitter.line(`const ${reshapeVar} = ${resultVar};`);
    }
    emitter.line(`const ${output} = builder.reshape(${reshapeVar}, [${outputShape.join(', ')}]);`);
  } else if (resultVar !== output) {
    emitter.line(`const ${output} = ${resultVar};`);
  }
}

function emitBatchMatMul(node: NodeIR, emitter: CodeEmitter): void {
  let a = emitter.ref(node.inputs[0]);
  let b = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);

  // WebNN matmul only supports float32/float16. Dequantize int8/uint8 inputs.
  a = emitDequantizeIfNeeded(a, node.inputs[0], 0, node, emitter, `${output}_a`);
  b = emitDequantizeIfNeeded(b, node.inputs[1], 1, node, emitter, `${output}_b`);

  emitter.line(`const ${output} = builder.matmul(${a}, ${b});`);
}

registerTfliteOp('FULLY_CONNECTED', emitFullyConnected);
registerTfliteOp('BATCH_MATMUL', emitBatchMatMul);
