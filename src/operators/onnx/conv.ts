// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/conv_op_builder.cc
// Maps ONNX Conv, ConvInteger → conv2d, ConvTranspose → convTranspose2d

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOps } from '../registry.js';

function emitConv(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const weight = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);

  const isConvInteger = node.opType === 'ConvInteger';
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

  // auto_pad handling — compute explicit padding since autoPad is not in the WebNN spec
  const autoPad = node.attributes.auto_pad as string | undefined;
  if (autoPad && autoPad !== 'NOTSET' && (autoPad === 'SAME_UPPER' || autoPad === 'SAME_LOWER')) {
    // ONNX NCHW: H=shape[2], W=shape[3]
    const inputShape = emitter.tensorShape(node.inputs[0]);
    const weightShape = emitter.constantShape(node.inputs[1]);
    if (inputShape && typeof inputShape[2] === 'number' && typeof inputShape[3] === 'number') {
      const inH = inputShape[2] as number;
      const inW = inputShape[3] as number;
      // For conv2d, kernel H/W at indices 2,3 (OIHW default)
      const kH = weightShape[2];
      const kW = weightShape[3];
      const sH = strides ? strides[0] : 1;
      const sW = strides ? strides[1] : 1;
      const dH = dilations ? dilations[0] : 1;
      const dW = dilations ? dilations[1] : 1;
      const effectiveKH = (kH - 1) * dH + 1;
      const effectiveKW = (kW - 1) * dW + 1;
      const outH = Math.ceil(inH / sH);
      const outW = Math.ceil(inW / sW);
      const totalPadH = Math.max(0, (outH - 1) * sH + effectiveKH - inH);
      const totalPadW = Math.max(0, (outW - 1) * sW + effectiveKW - inW);
      if (autoPad === 'SAME_UPPER') {
        const padTop = Math.floor(totalPadH / 2);
        const padBottom = totalPadH - padTop;
        const padLeft = Math.floor(totalPadW / 2);
        const padRight = totalPadW - padLeft;
        opts.push(`padding: [${padTop}, ${padBottom}, ${padLeft}, ${padRight}]`);
      } else {
        // SAME_LOWER: extra padding goes to beginning
        const padBottom = Math.floor(totalPadH / 2);
        const padTop = totalPadH - padBottom;
        const padRight = Math.floor(totalPadW / 2);
        const padLeft = totalPadW - padRight;
        opts.push(`padding: [${padTop}, ${padBottom}, ${padLeft}, ${padRight}]`);
      }
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

  // Bias (optional 3rd input) — ConvInteger uses input[2] as x_zero_point, not bias
  if (!isConvInteger) {
    const hasBias = node.inputs.length > 2 && node.inputs[2] !== '';
    if (hasBias) {
      const bias = emitter.ref(node.inputs[2]);
      opts.push(`bias: ${bias}`);
    }
  }

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';

  if (isConvInteger) {
    // ORT conv_op_builder.cc: ConvInteger decomposes to dequantizeLinear → conv2d → cast(int32)
    // WebNN conv2d only supports float32/float16 inputs, so we need to dequantize first.
    // DEVIATION from ORT: use cast+sub instead of dequantizeLinear for x, because x_zero_point
    // may come from DynamicQuantizeLinear on a tensor whose rank is absent from value_info.
    // cast+sub is equivalent when scale=1.0: dequantize(x, 1.0, zp) ≡ cast(x) − cast(zp).

    // Dequantize x: cast(x, float32) - cast(x_zero_point, float32)
    const hasXZp = node.inputs.length > 2 && node.inputs[2] !== '';
    const castX = `${output}_cast_x`;
    emitter.line(`const ${castX} = builder.cast(${input}, 'float32');`);
    const dequantX = `${output}_dequant_x`;
    if (hasXZp) {
      const xZpRaw = emitter.ref(node.inputs[2]);
      const castXZp = `${output}_cast_x_zp`;
      emitter.line(`const ${castXZp} = builder.cast(${xZpRaw}, 'float32');`);
      emitter.line(`const ${dequantX} = builder.sub(${castX}, ${castXZp});`);
    } else {
      emitter.line(`const ${dequantX} = ${castX};`);
    }

    // Dequantize w: dequantizeLinear(w, scale=1.0, w_zero_point)
    // w is a constant with a known static shape, so dequantizeLinear with rank-matched scale/zp works.
    // w_zero_point is input[3] (optional), per-channel if rank-1.
    const weightShape = emitter.tensorShape(node.inputs[1]);
    const weightRank = weightShape ? weightShape.length : 4;
    const wTargetShape = Array(weightRank).fill(1);
    let wZpRef: string;
    const hasWZp = node.inputs.length > 3 && node.inputs[3] !== '';
    if (hasWZp) {
      const wZpRaw = emitter.ref(node.inputs[3]);
      // Per-channel w_zero_point: if 1-D with C_out elements, reshape to [C_out, 1, 1, 1]
      // so that it broadcasts correctly along dim 0 (output channels) of OIHW weight.
      const wZpShape = emitter.isConstant(node.inputs[3]) ? emitter.constantShape(node.inputs[3]) : null;
      const wZpGraphShape = emitter.tensorShape(node.inputs[3]);
      const wZpRank = wZpShape ? wZpShape.length : (wZpGraphShape ? wZpGraphShape.length : 0);
      if (wZpRank === 1 && weightShape && typeof weightShape[0] === 'number') {
        wTargetShape[0] = weightShape[0]; // C_out at dim 0 of OIHW weight
      }
      wZpRef = `${output}_w_zp_reshaped`;
      emitter.line(`const ${wZpRef} = builder.reshape(${wZpRaw}, [${wTargetShape.join(', ')}]);`);
    } else {
      const inputDtype = emitter.tensorDataType(node.inputs[0]) ?? 'uint8';
      const arrayType = inputDtype === 'int8' ? 'Int8Array' : 'Uint8Array';
      wZpRef = `${output}_w_zp`;
      emitter.line(`const ${wZpRef} = builder.constant({dataType: '${inputDtype}', shape: [${wTargetShape.join(', ')}]}, new ${arrayType}([0]));`);
    }
    const wScaleRef = `${output}_w_scale`;
    emitter.line(`const ${wScaleRef} = builder.constant({dataType: 'float32', shape: [${wTargetShape.join(', ')}]}, new Float32Array([1]));`);
    const dequantW = `${output}_dequant_w`;
    emitter.line(`const ${dequantW} = builder.dequantizeLinear(${weight}, ${wScaleRef}, ${wZpRef});`);

    // Conv2d with float32 inputs
    const convOut = `${output}_conv`;
    emitter.line(`const ${convOut} = builder.${webnnOp}(${dequantX}, ${dequantW}${optsStr});`);

    // Cast output to int32
    emitter.line(`const ${output} = builder.cast(${convOut}, 'int32');`);
  } else {
    emitter.line(`const ${output} = builder.${webnnOp}(${input}, ${weight}${optsStr});`);
  }
}

registerOnnxOps(['Conv', 'ConvInteger', 'ConvTranspose'], emitConv);
