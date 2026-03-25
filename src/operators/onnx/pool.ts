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

  // auto_pad handling
  const autoPad = node.attributes.auto_pad as string | undefined;
  if (autoPad && autoPad !== 'NOTSET') {
    if (autoPad === 'SAME_UPPER') {
      opts.push(`autoPad: 'same-upper'`);
    } else if (autoPad === 'SAME_LOWER') {
      opts.push(`autoPad: 'same-lower'`);
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
