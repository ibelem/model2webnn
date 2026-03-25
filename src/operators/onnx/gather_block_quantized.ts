// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/gatherBlockQuantized_op_builder.cc
// ONNX GatherBlockQuantized → WebNN: dequantize + gather

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

function emitGatherBlockQuantized(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);     // quantized data
  const indices = emitter.ref(node.inputs[1]);   // gather indices
  const scales = emitter.ref(node.inputs[2]);    // quantization scales
  const output = emitter.declare(node.outputs[0]);

  const bits = (node.attributes.bits as number) ?? 4;
  const gatherAxis = (node.attributes.gather_axis as number) ?? 0;

  emitter.comment(`GatherBlockQuantized — bits=${bits}, axis=${gatherAxis}`);

  // Zero points
  const hasZeroPoints = node.inputs.length > 3 && node.inputs[3] !== '';
  let zpVar: string;
  if (hasZeroPoints) {
    zpVar = emitter.ref(node.inputs[3]);
  } else {
    const defaultZp = bits === 4 ? 0 : 128;
    zpVar = `${output}_default_zp`;
    emitter.line(`const ${zpVar} = builder.constant({dataType: 'uint8', shape: []}, new Uint8Array([${defaultZp}]));`);
  }

  // Step 1: Dequantize
  const dequantized = `${output}_dq`;
  emitter.line(`const ${dequantized} = builder.dequantizeLinear(${input}, { scale: ${scales}, zeroPoint: ${zpVar} });`);

  // Step 2: Gather
  emitter.line(`const ${output} = builder.gather(${dequantized}, ${indices}, { axis: ${gatherAxis} });`);
}

registerOnnxOp('GatherBlockQuantized', emitGatherBlockQuantized);
