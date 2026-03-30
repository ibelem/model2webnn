// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/pool_op_builder.cc
// Maps: AveragePool, MaxPool, GlobalAveragePool, GlobalMaxPool, LpPool

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

function emitPool(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  let webnnOp: string;
  switch (node.opType) {
    case 'AveragePool': webnnOp = 'averagePool2d'; break;
    case 'MaxPool': webnnOp = 'maxPool2d'; break;
    case 'GlobalAveragePool': webnnOp = 'averagePool2d'; break;
    case 'GlobalMaxPool': webnnOp = 'maxPool2d'; break;
    case 'LpPool': webnnOp = 'l2Pool2d'; break;
    default: webnnOp = 'averagePool2d';
  }

  const isGlobal = node.opType.startsWith('Global');

  const opts: string[] = [];

  if (!isGlobal) {
    const kernelShape = node.attributes.kernel_shape as number[] | undefined;
    if (kernelShape) {
      opts.push(`windowDimensions: [${kernelShape.join(', ')}]`);
    }
  }

  // Padding: ONNX [top, left, bottom, right] → WebNN [top, bottom, left, right]
  const pads = node.attributes.pads as number[] | undefined;
  if (pads && pads.length === 4 && pads.some((p) => p !== 0)) {
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

  // auto_pad handling — compute explicit padding since autoPad is not in the WebNN spec
  const autoPad = node.attributes.auto_pad as string | undefined;
  if (autoPad && autoPad !== 'NOTSET' && (autoPad === 'SAME_UPPER' || autoPad === 'SAME_LOWER')) {
    const inputShape = emitter.tensorShape(node.inputs[0]);
    const kernelShape = node.attributes.kernel_shape as number[] | undefined;
    if (inputShape && kernelShape && typeof inputShape[2] === 'number' && typeof inputShape[3] === 'number') {
      const inH = inputShape[2] as number;
      const inW = inputShape[3] as number;
      const kH = kernelShape[0];
      const kW = kernelShape[1];
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
        const padBottom = Math.floor(totalPadH / 2);
        const padTop = totalPadH - padBottom;
        const padRight = Math.floor(totalPadW / 2);
        const padLeft = totalPadW - padRight;
        opts.push(`padding: [${padTop}, ${padBottom}, ${padLeft}, ${padRight}]`);
      }
    }
  }

  // ceil_mode for non-global
  const ceilMode = (node.attributes.ceil_mode as number) ?? 0;
  if (ceilMode) {
    opts.push(`roundingType: 'ceil'`);
  }

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.${webnnOp}(${input}${optsStr});`);
}

registerOnnxOp('AveragePool', emitPool);
registerOnnxOp('MaxPool', emitPool);
registerOnnxOp('GlobalAveragePool', emitPool);
registerOnnxOp('GlobalMaxPool', emitPool);
registerOnnxOp('LpPool', emitPool);
