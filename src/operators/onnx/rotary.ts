// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/rotaryEmbedding_op_builder.cc
// ONNX RotaryEmbedding → WebNN decomposition
// Decomposes to: split → mul(cos) → mul(sin) → add → concat

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

function emitRotaryEmbedding(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);  // [B, S, N, H] or [B, S, hidden]
  const output = emitter.declare(node.outputs[0]);

  emitter.comment('RotaryEmbedding');

  // Input layout depends on domain — com.microsoft vs onnx
  // Determine cos/sin cache input indices
  // Standard: input(0), position_ids(1), cos_cache(2), sin_cache(3)
  // For opset 23+, position_ids may be optional
  const posIdsIdx = 1;
  const cosCacheIdx = 2;
  const sinCacheIdx = 3;

  const hasPosIds = node.inputs.length > posIdsIdx && node.inputs[posIdsIdx] !== '';
  const cosCache = emitter.ref(node.inputs[cosCacheIdx]);
  const sinCache = emitter.ref(node.inputs[sinCacheIdx]);

  // Gather cos/sin using position_ids
  let cos: string, sin: string;
  if (hasPosIds) {
    const posIds = emitter.ref(node.inputs[posIdsIdx]);
    cos = `${output}_cos`;
    sin = `${output}_sin`;
    emitter.line(`const ${cos} = builder.gather(${cosCache}, ${posIds}, { axis: 0 });`);
    emitter.line(`const ${sin} = builder.gather(${sinCache}, ${posIds}, { axis: 0 });`);
  } else {
    cos = cosCache;
    sin = sinCache;
  }

  // Split input into halves along last dimension
  const x1 = `${output}_x1`;
  const x2 = `${output}_x2`;
  emitter.line(`const ${x1} = builder.split(${input}, 2, { axis: -1 })[0];`);
  emitter.line(`const ${x2} = builder.split(${input}, 2, { axis: -1 })[1];`);

  // x' = x * cos - rotate_half(x) * sin
  // rotate_half(x) = [-x2, x1]
  const xCos = `${output}_xcos`;
  emitter.line(`const ${xCos} = builder.mul(${input}, ${cos});`);

  const negX2 = `${output}_negx2`;
  emitter.line(`const ${negX2} = builder.neg(${x2});`);
  const rotated = `${output}_rot`;
  emitter.line(`const ${rotated} = builder.concat([${negX2}, ${x1}], -1);`);
  const xSin = `${output}_xsin`;
  emitter.line(`const ${xSin} = builder.mul(${rotated}, ${sin});`);

  emitter.line(`const ${output} = builder.add(${xCos}, ${xSin});`);
}

registerOnnxOp('RotaryEmbedding', emitRotaryEmbedding);
