// TFLite resize ops: RESIZE_BILINEAR, RESIZE_NEAREST_NEIGHBOR

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';

/**
 * Emit resample2d for an NHWC tensor.
 * Uses axes: [1, 2] to resize spatial dimensions directly in NHWC layout.
 * The Chromium TFLite backend declares Resample2DAxes::kChannelsLast,
 * which means it only accepts axes [1, 2] — the spatial dims in NHWC.
 */
function emitNhwcResize(
  input: string,
  output: string,
  mode: 'linear' | 'nearest-neighbor',
  node: NodeIR,
  emitter: CodeEmitter,
): void {
  const opts: string[] = [`mode: '${mode}'`];

  // Extract target [height, width] from constant int32 tensor
  if (node.inputs.length > 1 && emitter.isConstant(node.inputs[1])) {
    const rawData = emitter.constantRawData(node.inputs[1]);
    if (rawData) {
      const aligned = new ArrayBuffer(rawData.byteLength);
      new Uint8Array(aligned).set(rawData);
      const int32View = new Int32Array(aligned);
      opts.push(`sizes: [${int32View[0]}, ${int32View[1]}]`);
    }
  }

  // Use axes [1, 2] for NHWC spatial dimensions (H, W)
  opts.push(`axes: [1, 2]`);

  emitter.line(`const ${output} = builder.resample2d(${input}, { ${opts.join(', ')} });`);
}

function emitResizeBilinear(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  emitter.comment('RESIZE_BILINEAR (NHWC axes [1,2])');
  emitNhwcResize(input, output, 'linear', node, emitter);
}

function emitResizeNearestNeighbor(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  emitter.comment('RESIZE_NEAREST_NEIGHBOR (NHWC axes [1,2])');
  emitNhwcResize(input, output, 'nearest-neighbor', node, emitter);
}

registerTfliteOp('RESIZE_BILINEAR', emitResizeBilinear);
registerTfliteOp('RESIZE_NEAREST_NEIGHBOR', emitResizeNearestNeighbor);
