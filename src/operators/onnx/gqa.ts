// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/gqa_op_builder.cc
// ONNX GroupQueryAttention → WebNN decomposition
// Decomposes to: [split QKV] → [rotary] → reshape → transpose → expand → scaledDotProductAttention

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

function emitGQA(node: NodeIR, emitter: CodeEmitter): void {
  const query = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const numHeads = (node.attributes.num_heads as number) ?? 1;
  const kvNumHeads = (node.attributes.kv_num_heads as number) ?? numHeads;
  const scale = node.attributes.scale as number | undefined;
  const doRotary = (node.attributes.do_rotary as number) ?? 0;

  const groupSize = numHeads / kvNumHeads;

  emitter.comment(`GroupQueryAttention — num_heads=${numHeads}, kv_num_heads=${kvNumHeads}, group_size=${groupSize}`);

  // Check if packed QKV (key/value are empty)
  const hasKey = node.inputs.length > 1 && node.inputs[1] !== '';
  const hasValue = node.inputs.length > 2 && node.inputs[2] !== '';
  const hasPastKey = node.inputs.length > 3 && node.inputs[3] !== '';
  const hasPastValue = node.inputs.length > 4 && node.inputs[4] !== '';
  const hasCosCache = node.inputs.length > 7 && node.inputs[7] !== '';
  const hasSinCache = node.inputs.length > 8 && node.inputs[8] !== '';

  let qVar: string, kVar: string, vVar: string;

  if (!hasKey || !hasValue) {
    // Packed QKV — split query [B, S, num_heads*H + 2*kv_num_heads*H]
    const splitVar = `${output}_qkv`;
    emitter.line(`const ${splitVar} = builder.split(${query}, 3, { axis: 2 });`);
    qVar = `${output}_q_packed`;
    kVar = `${output}_k_packed`;
    vVar = `${output}_v_packed`;
    emitter.line(`const ${qVar} = ${splitVar}[0];`);
    emitter.line(`const ${kVar} = ${splitVar}[1];`);
    emitter.line(`const ${vVar} = ${splitVar}[2];`);
  } else {
    qVar = query;
    kVar = emitter.ref(node.inputs[1]);
    vVar = emitter.ref(node.inputs[2]);
  }

  // Reshape to [B, S, N, H] then transpose to [B, N, S, H]
  const qReshaped = `${output}_q_r`;
  const kReshaped = `${output}_k_r`;
  const vReshaped = `${output}_v_r`;
  emitter.line(`const ${qReshaped} = builder.reshape(${qVar}, [0, 0, ${numHeads}, -1]);`);
  emitter.line(`const ${kReshaped} = builder.reshape(${kVar}, [0, 0, ${kvNumHeads}, -1]);`);
  emitter.line(`const ${vReshaped} = builder.reshape(${vVar}, [0, 0, ${kvNumHeads}, -1]);`);

  // Apply rotary embedding if do_rotary
  let qFinal: string, kFinal: string;
  if (doRotary && hasCosCache && hasSinCache) {
    const cosCache = emitter.ref(node.inputs[7]);
    const sinCache = emitter.ref(node.inputs[8]);

    emitter.comment('Apply rotary embedding');

    // Transpose to [B, N, S, H] for rotation
    const qT = `${output}_q_t`;
    const kT = `${output}_k_t`;
    emitter.line(`const ${qT} = builder.transpose(${qReshaped}, { permutation: [0, 2, 1, 3] });`);
    emitter.line(`const ${kT} = builder.transpose(${kReshaped}, { permutation: [0, 2, 1, 3] });`);

    // Gather cos/sin based on position
    let cosGathered = cosCache;
    let sinGathered = sinCache;
    if (node.inputs.length > 9 && node.inputs[9] !== '') {
      const posIds = emitter.ref(node.inputs[9]);
      cosGathered = `${output}_cos_g`;
      sinGathered = `${output}_sin_g`;
      emitter.line(`const ${cosGathered} = builder.gather(${cosCache}, ${posIds}, { axis: 0 });`);
      emitter.line(`const ${sinGathered} = builder.gather(${sinCache}, ${posIds}, { axis: 0 });`);
    }

    // Rotary: x' = x * cos - rot(x) * sin
    qFinal = applyRotary(`${output}_q_rot`, qT, cosGathered, sinGathered, emitter);
    kFinal = applyRotary(`${output}_k_rot`, kT, cosGathered, sinGathered, emitter);
  } else {
    qFinal = `${output}_q_trans`;
    kFinal = `${output}_k_trans`;
    emitter.line(`const ${qFinal} = builder.transpose(${qReshaped}, { permutation: [0, 2, 1, 3] });`);
    emitter.line(`const ${kFinal} = builder.transpose(${kReshaped}, { permutation: [0, 2, 1, 3] });`);
  }

  let vFinal = `${output}_v_trans`;
  emitter.line(`const ${vFinal} = builder.transpose(${vReshaped}, { permutation: [0, 2, 1, 3] });`);

  // Concat past KV
  if (hasPastKey) {
    const pastKey = emitter.ref(node.inputs[3]);
    const concatK = `${output}_k_cat`;
    emitter.line(`const ${concatK} = builder.concat([${pastKey}, ${kFinal}], 2);`);
    kFinal = concatK;
  }
  if (hasPastValue) {
    const pastValue = emitter.ref(node.inputs[4]);
    const concatV = `${output}_v_cat`;
    emitter.line(`const ${concatV} = builder.concat([${pastValue}, ${vFinal}], 2);`);
    vFinal = concatV;
  }

  // Expand KV heads for grouped attention
  if (groupSize > 1) {
    const kExpanded = `${output}_k_exp`;
    const vExpanded = `${output}_v_exp`;
    emitter.comment(`Expand KV for ${groupSize}x group size`);
    // Reshape [B, kv_heads, S, H] → [B, kv_heads, 1, S, H] → expand → [B, kv_heads, G, S, H] → reshape [B, num_heads, S, H]
    const kUnsq = `${output}_k_unsq`;
    const vUnsq = `${output}_v_unsq`;
    emitter.line(`const ${kUnsq} = builder.unsqueeze(${kFinal}, { axes: [2] });`);
    emitter.line(`const ${vUnsq} = builder.unsqueeze(${vFinal}, { axes: [2] });`);
    emitter.line(`const ${kExpanded} = builder.reshape(builder.expand(${kUnsq}, [0, 0, ${groupSize}, 0, 0]), [0, ${numHeads}, -1, 0]);`);
    emitter.line(`const ${vExpanded} = builder.reshape(builder.expand(${vUnsq}, [0, 0, ${groupSize}, 0, 0]), [0, ${numHeads}, -1, 0]);`);
    kFinal = kExpanded;
    vFinal = vExpanded;
  }

  // ScaledDotProductAttention
  const sdpaOpts: string[] = [];
  if (scale !== undefined) sdpaOpts.push(`scale: ${scale}`);
  const optsStr = sdpaOpts.length > 0 ? `, { ${sdpaOpts.join(', ')} }` : '';

  const attnOut = `${output}_attn`;
  emitter.line(`const ${attnOut} = builder.scaledDotProductAttention(${qFinal}, ${kFinal}, ${vFinal}${optsStr});`);

  // Transpose back and reshape
  const transBack = `${output}_trans_back`;
  emitter.line(`const ${transBack} = builder.transpose(${attnOut}, { permutation: [0, 2, 1, 3] });`);
  emitter.line(`const ${output} = builder.reshape(${transBack}, [0, 0, -1]);`);

  // Present key/value outputs
  if (node.outputs.length > 1 && node.outputs[1] !== '') {
    const presentKey = emitter.declare(node.outputs[1]);
    emitter.line(`const ${presentKey} = ${kFinal};`);
  }
  if (node.outputs.length > 2 && node.outputs[2] !== '') {
    const presentValue = emitter.declare(node.outputs[2]);
    emitter.line(`const ${presentValue} = ${vFinal};`);
  }
}

/** Apply rotary embedding: x' = x * cos - rotate_half(x) * sin */
function applyRotary(
  prefix: string,
  input: string,
  cos: string,
  sin: string,
  emitter: CodeEmitter,
): string {
  // Split into halves
  const x1 = `${prefix}_x1`;
  const x2 = `${prefix}_x2`;
  emitter.line(`const [${x1}, ${x2}] = [builder.split(${input}, 2, { axis: -1 })[0], builder.split(${input}, 2, { axis: -1 })[1]];`);

  // x * cos
  const xCos = `${prefix}_xcos`;
  emitter.line(`const ${xCos} = builder.mul(${input}, ${cos});`);

  // [-x2, x1] * sin
  const negX2 = `${prefix}_negx2`;
  emitter.line(`const ${negX2} = builder.neg(${x2});`);
  const rotated = `${prefix}_rotated`;
  emitter.line(`const ${rotated} = builder.concat([${negX2}, ${x1}], -1);`);
  const xSin = `${prefix}_xsin`;
  emitter.line(`const ${xSin} = builder.mul(${rotated}, ${sin});`);

  // Result
  const result = `${prefix}`;
  emitter.line(`const ${result} = builder.add(${xCos}, ${xSin});`);

  return result;
}

registerOnnxOp('GroupQueryAttention', emitGQA);
