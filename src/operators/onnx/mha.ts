// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/mha_op_builder.cc
// ONNX MultiHeadAttention → WebNN decomposition
// Decomposes to: reshape → transpose → [concat past KV] → scaledDotProductAttention → reshape

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

function emitMHA(node: NodeIR, emitter: CodeEmitter): void {
  const query = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const numHeads = (node.attributes.num_heads as number) ?? 1;
  const scale = node.attributes.scale as number | undefined;

  emitter.comment(`MultiHeadAttention — num_heads=${numHeads}`);

  // Determine if packed QKV (key/value inputs are empty)
  const hasKey = node.inputs.length > 1 && node.inputs[1] !== '';
  const hasValue = node.inputs.length > 2 && node.inputs[2] !== '';
  const hasPastKey = node.inputs.length > 3 && node.inputs[3] !== '';
  const hasPastValue = node.inputs.length > 4 && node.inputs[4] !== '';
  const hasAttentionBias = node.inputs.length > 5 && node.inputs[5] !== '';

  let qVar: string, kVar: string, vVar: string;

  if (!hasKey && !hasValue) {
    // Packed QKV — split along axis 3 if 5D [B, S, N, 3, H]
    const splitVar = `${output}_qkv_split`;
    emitter.line(`const ${splitVar} = builder.split(${query}, 3, { axis: 3 });`);
    const qSqueeze = `${output}_q`;
    const kSqueeze = `${output}_k`;
    const vSqueeze = `${output}_v`;
    emitter.line(`const ${qSqueeze} = builder.squeeze(${splitVar}[0], { axes: [3] });`);
    emitter.line(`const ${kSqueeze} = builder.squeeze(${splitVar}[1], { axes: [3] });`);
    emitter.line(`const ${vSqueeze} = builder.squeeze(${splitVar}[2], { axes: [3] });`);
    qVar = qSqueeze;
    kVar = kSqueeze;
    vVar = vSqueeze;
  } else {
    qVar = query;
    kVar = hasKey ? emitter.ref(node.inputs[1]) : query;
    vVar = hasValue ? emitter.ref(node.inputs[2]) : query;

    // Reshape Q, K, V from [B, S, hidden] to [B, S, N, H] then transpose to [B, N, S, H]
    const qReshaped = `${output}_q_r`;
    const kReshaped = `${output}_k_r`;
    const vReshaped = `${output}_v_r`;
    emitter.line(`const ${qReshaped} = builder.reshape(${qVar}, [0, 0, ${numHeads}, -1]);`);
    emitter.line(`const ${kReshaped} = builder.reshape(${kVar}, [0, 0, ${numHeads}, -1]);`);
    emitter.line(`const ${vReshaped} = builder.reshape(${vVar}, [0, 0, ${numHeads}, -1]);`);

    const qTransposed = `${output}_q_t`;
    const kTransposed = `${output}_k_t`;
    const vTransposed = `${output}_v_t`;
    emitter.line(`const ${qTransposed} = builder.transpose(${qReshaped}, { permutation: [0, 2, 1, 3] });`);
    emitter.line(`const ${kTransposed} = builder.transpose(${kReshaped}, { permutation: [0, 2, 1, 3] });`);
    emitter.line(`const ${vTransposed} = builder.transpose(${vReshaped}, { permutation: [0, 2, 1, 3] });`);

    qVar = qTransposed;
    kVar = kTransposed;
    vVar = vTransposed;
  }

  // Concat past_key/past_value with current K/V
  if (hasPastKey) {
    const pastKey = emitter.ref(node.inputs[3]);
    const concatK = `${output}_k_concat`;
    emitter.line(`const ${concatK} = builder.concat([${pastKey}, ${kVar}], 2);`);
    kVar = concatK;
  }
  if (hasPastValue) {
    const pastValue = emitter.ref(node.inputs[4]);
    const concatV = `${output}_v_concat`;
    emitter.line(`const ${concatV} = builder.concat([${pastValue}, ${vVar}], 2);`);
    vVar = concatV;
  }

  // ScaledDotProductAttention
  const sdpaOpts: string[] = [];
  if (scale !== undefined) {
    sdpaOpts.push(`scale: ${scale}`);
  }
  if (hasAttentionBias) {
    sdpaOpts.push(`mask: ${emitter.ref(node.inputs[5])}`);
  }

  const optsStr = sdpaOpts.length > 0 ? `, { ${sdpaOpts.join(', ')} }` : '';
  const attnOut = `${output}_attn`;
  emitter.line(`const ${attnOut} = builder.scaledDotProductAttention(${qVar}, ${kVar}, ${vVar}${optsStr});`);

  // Transpose back [B, N, S, H] → [B, S, N, H] and reshape to [B, S, hidden]
  const transBack = `${output}_trans`;
  emitter.line(`const ${transBack} = builder.transpose(${attnOut}, { permutation: [0, 2, 1, 3] });`);
  emitter.line(`const ${output} = builder.reshape(${transBack}, [0, 0, -1]);`);

  // Output present_key and present_value if needed
  if (node.outputs.length > 1 && node.outputs[1] !== '') {
    const presentKey = emitter.declare(node.outputs[1]);
    emitter.line(`const ${presentKey} = ${kVar};`);
  }
  if (node.outputs.length > 2 && node.outputs[2] !== '') {
    const presentValue = emitter.declare(node.outputs[2]);
    emitter.line(`const ${presentValue} = ${vVar};`);
  }
}

registerOnnxOp('MultiHeadAttention', emitMHA);
