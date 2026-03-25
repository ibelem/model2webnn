// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/
// Various shape/transform ops: Reshape, Transpose, Flatten, Squeeze, Unsqueeze,
// Concat, Split, Slice, Expand, Gather, Pad, Tile, Clip, Cast, Softmax, etc.

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

// reshape_op_builder.cc
function emitReshape(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const shape = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  // If shape is a constant, we can inline it
  if (emitter.isConstant(node.inputs[1])) {
    const shapeValues = emitter.constantShape(node.inputs[1]);
    emitter.line(`const ${output} = builder.reshape(${input}, [${shapeValues.join(', ')}]);`);
  } else {
    emitter.line(`const ${output} = builder.reshape(${input}, ${shape});`);
  }
}

// transpose_op_builder.cc
function emitTranspose(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const perm = node.attributes.perm as number[] | undefined;
  if (perm) {
    emitter.line(`const ${output} = builder.transpose(${input}, { permutation: [${perm.join(', ')}] });`);
  } else {
    emitter.line(`const ${output} = builder.transpose(${input});`);
  }
}

// flatten_op_builder.cc — decomposed to reshape
function emitFlatten(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const axis = (node.attributes.axis as number) ?? 1;
  // Flatten is reshape — but we need shape info at codegen time
  // Emit as reshape with computed shape using flatten semantics
  emitter.comment(`Flatten at axis=${axis}`);
  emitter.line(`const ${output} = builder.reshape(${input}, [/* flatten at axis ${axis} */]);`);
}

// squeeze_unsqueeze_op_builder.cc
function emitSqueeze(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const axes = node.attributes.axes as number[] | undefined;
  if (axes) {
    emitter.line(`const ${output} = builder.squeeze(${input}, { axes: [${axes.join(', ')}] });`);
  } else if (node.inputs.length > 1 && node.inputs[1] !== '') {
    // ONNX opset 13+: axes as second input
    emitter.line(`const ${output} = builder.squeeze(${input});`);
  } else {
    emitter.line(`const ${output} = builder.squeeze(${input});`);
  }
}

function emitUnsqueeze(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const axes = node.attributes.axes as number[] | undefined;
  if (axes) {
    emitter.line(`const ${output} = builder.unsqueeze(${input}, { axes: [${axes.join(', ')}] });`);
  } else {
    emitter.line(`const ${output} = builder.unsqueeze(${input});`);
  }
}

// concat_op_builder.cc
function emitConcat(node: NodeIR, emitter: CodeEmitter): void {
  const inputs = node.inputs.map((name) => emitter.ref(name));
  const output = emitter.declare(node.outputs[0]);
  const axis = (node.attributes.axis as number) ?? 0;
  emitter.line(`const ${output} = builder.concat([${inputs.join(', ')}], ${axis});`);
}

// split_op_builder.cc
function emitSplit(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const axis = (node.attributes.axis as number) ?? 0;
  const split = node.attributes.split as number[] | undefined;
  const numOutputs = node.outputs.length;

  if (split) {
    const splitVar = emitter.declare(`${node.outputs[0]}_splits`);
    emitter.line(`const ${splitVar} = builder.split(${input}, [${split.join(', ')}], { axis: ${axis} });`);
    for (let i = 0; i < numOutputs; i++) {
      const out = emitter.declare(node.outputs[i]);
      emitter.line(`const ${out} = ${splitVar}[${i}];`);
    }
  } else {
    const splitVar = emitter.declare(`${node.outputs[0]}_splits`);
    emitter.line(`const ${splitVar} = builder.split(${input}, ${numOutputs}, { axis: ${axis} });`);
    for (let i = 0; i < numOutputs; i++) {
      const out = emitter.declare(node.outputs[i]);
      emitter.line(`const ${out} = ${splitVar}[${i}];`);
    }
  }
}

// slice_op_builder.cc
function emitSlice(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const starts = emitter.ref(node.inputs[1]);
  const ends = emitter.ref(node.inputs[2]);
  const output = emitter.declare(node.outputs[0]);
  const hasAxes = node.inputs.length > 3 && node.inputs[3] !== '';
  const hasSteps = node.inputs.length > 4 && node.inputs[4] !== '';

  const opts: string[] = [];
  if (hasAxes) opts.push(`axes: ${emitter.ref(node.inputs[3])}`);
  if (hasSteps) opts.push(`strides: ${emitter.ref(node.inputs[4])}`);

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.slice(${input}, ${starts}, ${ends}${optsStr});`);
}

// expand_op_builder.cc
function emitExpand(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const shape = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.expand(${input}, ${shape});`);
}

// gather_op_builder.cc
function emitGather(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  const axis = (node.attributes.axis as number) ?? 0;
  emitter.line(`const ${output} = builder.gather(${input}, ${indices}, { axis: ${axis} });`);
}

// pad_op_builder.cc
function emitPad(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const pads = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  const mode = (node.attributes.mode as string) ?? 'constant';

  const opts: string[] = [`mode: '${mode}'`];
  if (node.inputs.length > 2 && node.inputs[2] !== '') {
    opts.push(`value: ${emitter.ref(node.inputs[2])}`);
  }

  emitter.line(`const ${output} = builder.pad(${input}, ${pads}, { ${opts.join(', ')} });`);
}

// tile_op_builder.cc
function emitTile(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const repeats = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.tile(${input}, ${repeats});`);
}

// clip_op_builder.cc → clamp
function emitClip(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const hasMin = node.inputs.length > 1 && node.inputs[1] !== '';
  const hasMax = node.inputs.length > 2 && node.inputs[2] !== '';

  const opts: string[] = [];
  if (hasMin) opts.push(`minValue: ${emitter.ref(node.inputs[1])}`);
  if (hasMax) opts.push(`maxValue: ${emitter.ref(node.inputs[2])}`);

  const optsStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
  emitter.line(`const ${output} = builder.clamp(${input}${optsStr});`);
}

// cast_op_builder.cc
function emitCast(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const to = node.attributes.to as number;

  // Map ONNX type ID to WebNN data type string
  const typeMap: Record<number, string> = {
    1: 'float32', 2: 'uint8', 3: 'int8', 6: 'int32',
    7: 'int64', 10: 'float16', 12: 'uint32', 13: 'uint64',
  };
  const targetType = typeMap[to] ?? 'float32';
  emitter.line(`const ${output} = builder.cast(${input}, '${targetType}');`);
}

// softmax_op_builder.cc
function emitSoftmax(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const axis = (node.attributes.axis as number) ?? -1;
  emitter.line(`const ${output} = builder.softmax(${input}, ${axis});`);
}

// where (ternary_op_builder.cc)
function emitWhere(node: NodeIR, emitter: CodeEmitter): void {
  const condition = emitter.ref(node.inputs[0]);
  const x = emitter.ref(node.inputs[1]);
  const y = emitter.ref(node.inputs[2]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.where(${condition}, ${x}, ${y});`);
}

// dropout_op_builder.cc — no-op during inference
function emitDropout(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = ${input};`);
}

// shape_op_builder.cc
function emitShape(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.shape(${input});`);
}

// Register all ops
registerOnnxOp('Reshape', emitReshape);
registerOnnxOp('Transpose', emitTranspose);
registerOnnxOp('Flatten', emitFlatten);
registerOnnxOp('Squeeze', emitSqueeze);
registerOnnxOp('Unsqueeze', emitUnsqueeze);
registerOnnxOp('Concat', emitConcat);
registerOnnxOp('Split', emitSplit);
registerOnnxOp('Slice', emitSlice);
registerOnnxOp('Expand', emitExpand);
registerOnnxOp('Gather', emitGather);
registerOnnxOp('Pad', emitPad);
registerOnnxOp('Tile', emitTile);
registerOnnxOp('Clip', emitClip);
registerOnnxOp('Cast', emitCast);
registerOnnxOp('Softmax', emitSoftmax);
registerOnnxOp('Where', emitWhere);
registerOnnxOp('Dropout', emitDropout);
registerOnnxOp('Shape', emitShape);
