// TFLite pooling ops: AVERAGE_POOL_2D, MAX_POOL_2D, L2_POOL_2D

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';
import { emitFusedActivation, computeSamePadding, emitDequantizeIfNeeded } from './common.js';

function emitPool2D(node: NodeIR, emitter: CodeEmitter): void {
  let input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  // WebNN pool ops require float32/float16 — dequantize if needed
  input = emitDequantizeIfNeeded(input, node.inputs[0], 0, node, emitter, `${output}_in`);

  const attrs = node.attributes;
  const windowDims: number[] = [Number(attrs.filter_height ?? 1), Number(attrs.filter_width ?? 1)];
  const strides: number[] = [Number(attrs.stride_h ?? 1), Number(attrs.stride_w ?? 1)];
  const padding = attrs.padding as string;

  const webnnOp = node.opType === 'MAX_POOL_2D' ? 'maxPool2d' : 'averagePool2d';

  emitter.comment(`${node.opType}`);

  const opts: string[] = [];
  opts.push(`windowDimensions: [${windowDims}]`);
  opts.push(`strides: [${strides}]`);
  if (padding === 'SAME') {
    const inputShape = emitter.tensorShape(node.inputs[0]);
    if (inputShape && typeof inputShape[1] === 'number' && typeof inputShape[2] === 'number') {
      const [padTop, padBottom, padLeft, padRight] = computeSamePadding(
        inputShape[1] as number, inputShape[2] as number,
        windowDims[0], windowDims[1],
        strides[0], strides[1], 1, 1);
      opts.push(`padding: [${padTop}, ${padBottom}, ${padLeft}, ${padRight}]`);
    }
  }
  opts.push(`layout: 'nhwc'`);

  const rawVar = `${output}_raw`;
  emitter.line(`const ${rawVar} = builder.${webnnOp}(${input}, { ${opts.join(', ')} });`);

  const activation = attrs.fused_activation as string | undefined;
  const resultVar = emitFusedActivation(rawVar, activation, emitter);

  if (resultVar !== output) {
    emitter.line(`const ${output} = ${resultVar};`);
  }
}

function emitL2Pool2D(node: NodeIR, emitter: CodeEmitter): void {
  let input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  input = emitDequantizeIfNeeded(input, node.inputs[0], 0, node, emitter, `${output}_in`);

  const attrs = node.attributes;
  const windowDims: number[] = [Number(attrs.filter_height ?? 1), Number(attrs.filter_width ?? 1)];
  const strides: number[] = [Number(attrs.stride_h ?? 1), Number(attrs.stride_w ?? 1)];
  const padding = attrs.padding as string;

  emitter.comment('L2_POOL_2D');

  const opts: string[] = [];
  opts.push(`windowDimensions: [${windowDims}]`);
  opts.push(`strides: [${strides}]`);
  if (padding === 'SAME') {
    const inputShape = emitter.tensorShape(node.inputs[0]);
    if (inputShape && typeof inputShape[1] === 'number' && typeof inputShape[2] === 'number') {
      const [padTop, padBottom, padLeft, padRight] = computeSamePadding(
        inputShape[1] as number, inputShape[2] as number,
        windowDims[0], windowDims[1],
        strides[0], strides[1], 1, 1);
      opts.push(`padding: [${padTop}, ${padBottom}, ${padLeft}, ${padRight}]`);
    }
  }
  opts.push(`layout: 'nhwc'`);

  emitter.line(`const ${output} = builder.l2Pool2d(${input}, { ${opts.join(', ')} });`);
}

registerTfliteOp('AVERAGE_POOL_2D', emitPool2D);
registerTfliteOp('MAX_POOL_2D', emitPool2D);
registerTfliteOp('L2_POOL_2D', emitL2Pool2D);
