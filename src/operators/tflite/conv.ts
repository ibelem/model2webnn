// TFLite convolution ops: CONV_2D, DEPTHWISE_CONV_2D, TRANSPOSE_CONV

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';
import { emitFusedActivation, computeSamePadding, emitDequantizeIfNeeded } from './common.js';

function emitConv2D(node: NodeIR, emitter: CodeEmitter): void {
  let input = emitter.ref(node.inputs[0]);
  let filter = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  let bias = node.inputs.length > 2 ? emitter.ref(node.inputs[2]) : undefined;

  // WebNN conv2d requires float32/float16 — dequantize quantized inputs
  input = emitDequantizeIfNeeded(input, node.inputs[0], 0, node, emitter, `${output}_in`);
  filter = emitDequantizeIfNeeded(filter, node.inputs[1], 1, node, emitter, `${output}_filt`);
  if (bias) bias = emitDequantizeIfNeeded(bias, node.inputs[2], 2, node, emitter, `${output}_bias`);

  const attrs = node.attributes;
  const strides: number[] = [Number(attrs.stride_h ?? 1), Number(attrs.stride_w ?? 1)];
  const dilations: number[] = [Number(attrs.dilation_h_factor ?? 1), Number(attrs.dilation_w_factor ?? 1)];
  const padding = attrs.padding as string;

  emitter.comment(`Conv2D — padding: ${padding}`);

  const opts: string[] = [];
  opts.push(`strides: [${strides}]`);
  opts.push(`dilations: [${dilations}]`);
  const inputShape = emitter.tensorShape(node.inputs[0]);
  const filterShape = emitter.constantShape(node.inputs[1]); // OHWI: [outC, kH, kW, inC_per_group]
  if (padding === 'SAME') {
    // Compute explicit padding — autoPad is not in the WebNN spec
    if (inputShape && typeof inputShape[1] === 'number' && typeof inputShape[2] === 'number') {
      const [padTop, padBottom, padLeft, padRight] = computeSamePadding(
        inputShape[1] as number, inputShape[2] as number,
        filterShape[1], filterShape[2],
        strides[0], strides[1], dilations[0], dilations[1]);
      opts.push(`padding: [${padTop}, ${padBottom}, ${padLeft}, ${padRight}]`);
    }
  }
  opts.push(`inputLayout: 'nhwc'`);
  opts.push(`filterLayout: 'ohwi'`);
  // Detect grouped convolution: if filter inC_per_group < input channels
  if (inputShape && typeof inputShape[3] === 'number' && filterShape.length === 4) {
    const inC = inputShape[3] as number;
    const filterInC = filterShape[3]; // inC per group
    if (filterInC > 0 && inC > filterInC && inC % filterInC === 0) {
      opts.push(`groups: ${inC / filterInC}`);
    }
  }
  if (bias) {
    opts.push(`bias: ${bias}`);
  }

  const rawVar = `${output}_raw`;
  emitter.line(`const ${rawVar} = builder.conv2d(${input}, ${filter}, { ${opts.join(', ')} });`);

  const activation = attrs.fused_activation as string | undefined;
  const resultVar = emitFusedActivation(rawVar, activation, emitter);

  if (resultVar !== output) {
    emitter.line(`const ${output} = ${resultVar};`);
  }
}

function emitDepthwiseConv2D(node: NodeIR, emitter: CodeEmitter): void {
  let input = emitter.ref(node.inputs[0]);
  let filter = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  let bias = node.inputs.length > 2 ? emitter.ref(node.inputs[2]) : undefined;

  // WebNN conv2d requires float32/float16 — dequantize quantized inputs
  input = emitDequantizeIfNeeded(input, node.inputs[0], 0, node, emitter, `${output}_in`);
  filter = emitDequantizeIfNeeded(filter, node.inputs[1], 1, node, emitter, `${output}_filt`);
  if (bias) bias = emitDequantizeIfNeeded(bias, node.inputs[2], 2, node, emitter, `${output}_bias`);

  const attrs = node.attributes;
  const strides: number[] = [Number(attrs.stride_h ?? 1), Number(attrs.stride_w ?? 1)];
  const dilations: number[] = [Number(attrs.dilation_h_factor ?? 1), Number(attrs.dilation_w_factor ?? 1)];
  const padding = attrs.padding as string;

  emitter.comment(`DepthwiseConv2D — padding: ${padding}`);

  const opts: string[] = [];
  opts.push(`strides: [${strides}]`);
  opts.push(`dilations: [${dilations}]`);
  if (padding === 'SAME') {
    // Compute explicit padding — autoPad is not in the WebNN spec
    const inputShape = emitter.tensorShape(node.inputs[0]);
    const filterShape2 = emitter.constantShape(node.inputs[1]); // IHWO: [inC, kH, kW, outC]
    if (inputShape && typeof inputShape[1] === 'number' && typeof inputShape[2] === 'number') {
      const [padTop, padBottom, padLeft, padRight] = computeSamePadding(
        inputShape[1] as number, inputShape[2] as number,
        filterShape2[1], filterShape2[2],
        strides[0], strides[1], dilations[0], dilations[1]);
      opts.push(`padding: [${padTop}, ${padBottom}, ${padLeft}, ${padRight}]`);
    }
  }
  opts.push(`inputLayout: 'nhwc'`);
  opts.push(`filterLayout: 'ihwo'`);
  // groups = output channels for depthwise (last dim of IHWO filter)
  const filterShape = emitter.constantShape(node.inputs[1]);
  if (filterShape.length === 4 && filterShape[3] > 0) {
    opts.push(`groups: ${filterShape[3]}`);
  } else {
    opts.push(`groups: ${filter}.shape[3]`);
  }
  if (bias) {
    opts.push(`bias: ${bias}`);
  }

  const rawVar = `${output}_raw`;
  emitter.line(`const ${rawVar} = builder.conv2d(${input}, ${filter}, { ${opts.join(', ')} });`);

  const activation = attrs.fused_activation as string | undefined;
  const resultVar = emitFusedActivation(rawVar, activation, emitter);

  if (resultVar !== output) {
    emitter.line(`const ${output} = ${resultVar};`);
  }
}

function emitTransposeConv(node: NodeIR, emitter: CodeEmitter): void {
  // TFLite TRANSPOSE_CONV inputs: output_shape, filter, input [, bias]
  let filter = emitter.ref(node.inputs[1]);
  let input = emitter.ref(node.inputs[2]);
  const output = emitter.declare(node.outputs[0]);
  let bias = node.inputs.length > 3 ? emitter.ref(node.inputs[3]) : undefined;

  // WebNN convTranspose2d requires float32/float16 — dequantize quantized inputs
  input = emitDequantizeIfNeeded(input, node.inputs[2], 2, node, emitter, `${output}_in`);
  filter = emitDequantizeIfNeeded(filter, node.inputs[1], 1, node, emitter, `${output}_filt`);
  if (bias) bias = emitDequantizeIfNeeded(bias, node.inputs[3], 3, node, emitter, `${output}_bias`);

  const attrs = node.attributes;
  const strides: number[] = [Number(attrs.stride_h ?? 1), Number(attrs.stride_w ?? 1)];
  const padding = attrs.padding as string;

  emitter.comment('TransposeConv');

  const opts: string[] = [];
  opts.push(`strides: [${strides}]`);
  if (padding === 'SAME') {
    // Compute explicit padding for transpose conv
    // TFLite inputs[0] is the output shape, inputs[2] is the input
    const inputShape = emitter.tensorShape(node.inputs[2]);
    const filterShape = emitter.constantShape(node.inputs[1]); // OHWI: [outC, kH, kW, inC]
    if (inputShape && typeof inputShape[1] === 'number' && typeof inputShape[2] === 'number') {
      const inH = inputShape[1] as number;
      const inW = inputShape[2] as number;
      const kH = filterShape[1];
      const kW = filterShape[2];
      // For convTranspose2d: outputSize = (inputSize-1)*stride + filterSize - padBegin - padEnd
      // TFLite SAME: outputSize = inputSize * stride
      const totalPadH = Math.max(0, (inH - 1) * strides[0] + kH - inH * strides[0]);
      const totalPadW = Math.max(0, (inW - 1) * strides[1] + kW - inW * strides[1]);
      const padTop = Math.floor(totalPadH / 2);
      const padBottom = totalPadH - padTop;
      const padLeft = Math.floor(totalPadW / 2);
      const padRight = totalPadW - padLeft;
      opts.push(`padding: [${padTop}, ${padBottom}, ${padLeft}, ${padRight}]`);
    }
  }
  opts.push(`inputLayout: 'nhwc'`);
  opts.push(`filterLayout: 'ohwi'`);
  if (bias) {
    opts.push(`bias: ${bias}`);
  }

  emitter.line(`const ${output} = builder.convTranspose2d(${input}, ${filter}, { ${opts.join(', ')} });`);
}

registerTfliteOp('CONV_2D', emitConv2D);
registerTfliteOp('DEPTHWISE_CONV_2D', emitDepthwiseConv2D);
registerTfliteOp('TRANSPOSE_CONV', emitTransposeConv);
