// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/conv_op_builder.cc
// Maps ONNX Conv, ConvInteger → conv2d, ConvTranspose → convTranspose2d

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOps } from '../registry.js';

function emitConv(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const weight = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);

  const isTranspose = node.opType === 'ConvTranspose';
  const webnnOp = isTranspose ? 'convTranspose2d' : 'conv2d';

  // Build options object
  const opts: string[] = [];

  // Padding: ONNX uses [top, left, bottom, right], WebNN uses [top, bottom, left, right]
  const pads = node.attributes.pads as number[] | undefined;
  if (pads && pads.length === 4) {
    // ONNX: [top, left, bottom, right] → WebNN: [top, bottom, left, right]
    opts.push(`padding: [${pads[0]}, ${pads[2]}, ${pads[1]}, ${pads[3]}]`);
  }

  const strides = node.attributes.strides as number[] | undefined;
  if (strides && !(strides[0] === 1 && strides[1] === 1)) {
    opts.push(`strides: [${strides.join(', ')}]`);
  }

  const dilations = node.attributes.dilations as number[] | undefined;
  if (dilations && !(dilations[0] === 1 && dilations[1] === 1)) {
    opts.push(`dilations: [${dilations.join(', ')}]`);
  }

  const group = (node.attributes.group as number) ?? 1;
  if (group > 1) {
    opts.push(`groups: ${group}`);
  }

  // auto_pad handling
  const autoPad = node.attributes.auto_pad as string | undefined;
  if (autoPad && autoPad !== 'NOTSET') {
    if (autoPad === 'SAME_UPPER') {
      opts.push(`autoPad: 'same-upper'`);
    } else if (autoPad === 'SAME_LOWER') {
      opts.push(`autoPad: 'same-lower'`);
    }
  }

  // ConvTranspose specific: output_padding
  if (isTranspose) {
    const outputPadding = node.attributes.output_padding as number[] | undefined;
    if (outputPadding && outputPadding.some((p: number) => p !== 0)) {
      opts.push(`outputPadding: [${outputPadding.join(', ')}]`);
    }
    const outputShape = node.attributes.output_shape as number[] | undefined;
    if (outputShape) {
      opts.push(`outputSizes: [${outputShape.join(', ')}]`);
    }
  }

  // Bias (optional 3rd input)
  const hasBias = node.inputs.length > 2 && node.inputs[2] !== '';
  if (hasBias) {
    const bias = emitter.ref(node.inputs[2]);
    opts.push(`bias: ${bias}`);
  }

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.${webnnOp}(${input}, ${weight}${optsStr});`);
}

registerOnnxOps(['Conv', 'ConvInteger', 'ConvTranspose'], emitConv);
