// TFLite shape manipulation ops: RESHAPE, SQUEEZE, EXPAND_DIMS, TRANSPOSE,
// CONCATENATION, SPLIT, SPLIT_V, SLICE, STRIDED_SLICE, PAD, PADV2, MIRROR_PAD,
// TILE, GATHER, GATHER_ND, SCATTER_ND, SHAPE, PACK, UNPACK

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerTfliteOp } from '../registry.js';

function emitReshape(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  // Resolve shape values, replacing any -1 with the correct dimension.
  // WebNN reshape does not support -1; all dims must be unsigned long.
  function resolveShape(shape: number[]): number[] {
    const negIdx = shape.indexOf(-1);
    if (negIdx === -1) return shape;
    // Use output tensor shape from TFLite metadata
    const outShape = emitter.tensorShape(node.outputs[0]);
    if (outShape && outShape.every(d => typeof d === 'number' && d >= 0)) {
      return outShape as number[];
    }
    // Fallback: compute -1 dim from input total size
    const inShape = emitter.tensorShape(node.inputs[0]);
    if (inShape && inShape.every(d => typeof d === 'number')) {
      const totalSize = (inShape as number[]).reduce((a, b) => a * b, 1);
      const knownSize = shape.reduce((a, b) => b >= 0 ? a * b : a, 1);
      const resolved = [...shape];
      resolved[negIdx] = totalSize / knownSize;
      return resolved;
    }
    return shape;
  }

  // TFLite: new_shape comes from options or second input tensor
  const newShape = node.attributes.new_shape as number[] | undefined;
  if (newShape) {
    const resolved = resolveShape(newShape);
    emitter.line(`const ${output} = builder.reshape(${input}, [${resolved}]);`);
  } else if (node.inputs.length > 1) {
    // Shape is a constant int32 tensor — extract as literal array
    const shapeValues = emitter.constantIntValues(node.inputs[1]);
    if (shapeValues) {
      const resolved = resolveShape(shapeValues);
      emitter.line(`const ${output} = builder.reshape(${input}, [${resolved}]);`);
    } else {
      const shapeInput = emitter.ref(node.inputs[1]);
      emitter.line(`const ${output} = builder.reshape(${input}, ${shapeInput});`);
    }
  } else {
    emitter.comment('Reshape: no shape info available');
    emitter.line(`const ${output} = ${input};`);
  }
}

function emitSqueeze(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  // WebNN has no squeeze — use reshape with the known output shape
  const outShape = emitter.tensorShape(node.outputs[0]);
  if (outShape) {
    emitter.line(`const ${output} = builder.reshape(${input}, [${outShape.join(', ')}]);`);
  } else {
    // Fallback: compute from squeeze_dims attribute
    const squeezeDims = node.attributes.squeeze_dims as number[] | undefined;
    const inShape = emitter.tensorShape(node.inputs[0]);
    if (inShape && squeezeDims && squeezeDims.length > 0) {
      const dimSet = new Set(squeezeDims);
      const newShape = inShape.filter((_, i) => !dimSet.has(i));
      emitter.line(`const ${output} = builder.reshape(${input}, [${newShape.join(', ')}]);`);
    } else if (inShape) {
      const newShape = inShape.filter((d) => d !== 1);
      emitter.line(`const ${output} = builder.reshape(${input}, [${newShape.join(', ')}]);`);
    } else {
      emitter.comment('WARNING: unknown shape for squeeze');
      emitter.line(`const ${output} = ${input};`);
    }
  }
}

function emitExpandDims(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  // WebNN has no unsqueeze — use reshape with the known output shape
  const outShape = emitter.tensorShape(node.outputs[0]);
  if (outShape) {
    emitter.line(`const ${output} = builder.reshape(${input}, [${outShape.join(', ')}]);`);
  } else {
    // Fallback: insert a dim-1 at the specified axis
    const axisValues = node.inputs.length > 1 ? emitter.constantIntValues(node.inputs[1]) : null;
    const axis = axisValues ? axisValues[0] : 0;
    const inShape = emitter.tensorShape(node.inputs[0]);
    if (inShape) {
      const newShape = [...inShape];
      newShape.splice(axis < 0 ? inShape.length + 1 + axis : axis, 0, 1);
      emitter.line(`const ${output} = builder.reshape(${input}, [${newShape.join(', ')}]);`);
    } else {
      emitter.comment('WARNING: unknown shape for expand_dims');
      emitter.line(`const ${output} = ${input};`);
    }
  }
}

function emitTranspose(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  if (node.inputs.length > 1) {
    // Permutation is a constant int32 tensor — extract as literal array
    const permValues = emitter.constantIntValues(node.inputs[1]);
    if (permValues) {
      emitter.line(`const ${output} = builder.transpose(${input}, { permutation: [${permValues}] });`);
    } else {
      emitter.line(`const ${output} = builder.transpose(${input});`);
    }
  } else {
    emitter.line(`const ${output} = builder.transpose(${input});`);
  }
}

function emitConcatenation(node: NodeIR, emitter: CodeEmitter): void {
  const inputs = node.inputs.map((n) => emitter.ref(n));
  const output = emitter.declare(node.outputs[0]);
  let axis = (node.attributes.axis as number) ?? 0;
  // WebNN concat axis is unsigned long — normalize negative values
  if (axis < 0) {
    const inputShape = emitter.tensorShape(node.inputs[0]);
    if (inputShape) axis += inputShape.length;
  }
  emitter.line(`const ${output} = builder.concat([${inputs.join(', ')}], ${axis});`);
}

function emitSplit(node: NodeIR, emitter: CodeEmitter): void {
  // TFLite SPLIT: input(0) = split axis tensor, input(1) = data
  // The axis is from the first input (constant), num_splits from options
  const data = emitter.ref(node.inputs[1]);
  const numSplits = (node.attributes.num_splits as number) ?? node.outputs.length;

  // Axis is a scalar constant int32 tensor
  const axisValues = emitter.constantIntValues(node.inputs[0]);
  let axis = axisValues ? axisValues[0] : 0;
  // WebNN split axis is unsigned long — normalize negative values
  if (axis < 0) {
    const dataShape = emitter.tensorShape(node.inputs[1]);
    if (dataShape) axis += dataShape.length;
  }

  emitter.comment(`SPLIT into ${numSplits} parts along axis ${axis}`);

  // Emit split outputs
  for (let i = 0; i < node.outputs.length; i++) {
    const out = emitter.declare(node.outputs[i]);
    emitter.line(`const ${out} = builder.split(${data}, ${numSplits}, { axis: ${axis} })[${i}];`);
  }
}

function emitSplitV(node: NodeIR, emitter: CodeEmitter): void {
  // TFLite SPLIT_V: input(0) = data, input(1) = size_splits, input(2) = axis
  const data = emitter.ref(node.inputs[0]);

  // Read axis from third input (constant int32 scalar)
  let axis = 0;
  if (node.inputs.length > 2) {
    const axisValues = emitter.constantIntValues(node.inputs[2]);
    if (axisValues) {
      axis = axisValues[0];
      if (axis < 0) {
        const dataShape = emitter.tensorShape(node.inputs[0]);
        if (dataShape) axis += dataShape.length;
      }
    }
  }

  // Read size_splits from second input (constant int32 tensor)
  const sizeSplits = emitter.constantIntValues(node.inputs[1]);

  emitter.comment(`SPLIT_V into ${node.outputs.length} parts along axis ${axis}`);

  if (sizeSplits && sizeSplits.length === node.outputs.length) {
    // Use explicit sizes with WebNN split(input, splits_as_array, {axis})
    for (let i = 0; i < node.outputs.length; i++) {
      const out = emitter.declare(node.outputs[i]);
      emitter.line(`const ${out} = builder.split(${data}, [${sizeSplits}], { axis: ${axis} })[${i}];`);
    }
  } else {
    // Fallback: equal split
    for (let i = 0; i < node.outputs.length; i++) {
      const out = emitter.declare(node.outputs[i]);
      emitter.line(`const ${out} = builder.split(${data}, ${node.outputs.length}, { axis: ${axis} })[${i}];`);
    }
  }
}

function emitSlice(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // TFLite SLICE: inputs are data, begin, size — these are constant int32 tensors
  // Size value of -1 means "to the end" of that dimension
  // WebNN slice requires positive unsigned long sizes
  if (node.inputs.length >= 3) {
    const beginVals = emitter.constantIntValues(node.inputs[1]);
    const sizeVals = emitter.constantIntValues(node.inputs[2]);
    if (beginVals && sizeVals) {
      // Resolve -1 sizes using input shape
      const resolvedSizes = [...sizeVals];
      const inputShape = emitter.tensorShape(node.inputs[0]);
      for (let i = 0; i < resolvedSizes.length; i++) {
        if (resolvedSizes[i] === -1 && inputShape && typeof inputShape[i] === 'number') {
          resolvedSizes[i] = (inputShape[i] as number) - beginVals[i];
        }
      }
      emitter.line(`const ${output} = builder.slice(${input}, [${beginVals}], [${resolvedSizes}]);`);
    } else {
      // Fallback for non-constant begin/size
      const begin = emitter.ref(node.inputs[1]);
      const size = emitter.ref(node.inputs[2]);
      emitter.line(`const ${output} = builder.slice(${input}, ${begin}, ${size});`);
    }
  } else {
    emitter.line(`const ${output} = ${input}; // SLICE (missing begin/size)`);
  }
}

function emitStridedSlice(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  emitter.comment('STRIDED_SLICE');

  // Read begin, end, strides from constant tensors
  const beginVals = node.inputs.length > 1 ? emitter.constantIntValues(node.inputs[1]) : null;
  const endVals = node.inputs.length > 2 ? emitter.constantIntValues(node.inputs[2]) : null;
  const strideVals = node.inputs.length > 3 ? emitter.constantIntValues(node.inputs[3]) : null;
  const inputShape = emitter.tensorShape(node.inputs[0]);

  if (!beginVals || !endVals || !inputShape) {
    // Fallback: use ref-based slice (can't resolve constants)
    if (node.inputs.length >= 3) {
      const begin = emitter.ref(node.inputs[1]);
      const end = emitter.ref(node.inputs[2]);
      emitter.line(`const ${output} = builder.slice(${input}, ${begin}, ${end});`);
    } else {
      emitter.line(`const ${output} = ${input}; // STRIDED_SLICE (missing begin/end)`);
    }
    return;
  }

  const rank = inputShape.length;
  const beginMask = (node.attributes.begin_mask as number) ?? 0;
  const endMask = (node.attributes.end_mask as number) ?? 0;
  const shrinkMask = (node.attributes.shrink_axis_mask as number) ?? 0;

  // Compute effective starts and sizes for each dimension
  const starts: number[] = [];
  const sizes: number[] = [];
  const strides: number[] = [];
  const shrinkAxes: number[] = [];

  for (let i = 0; i < rank; i++) {
    const dimSize = typeof inputShape[i] === 'number' ? (inputShape[i] as number) : -1;
    const stride = strideVals && i < strideVals.length ? strideVals[i] : 1;
    strides.push(stride);

    let start = i < beginVals.length ? beginVals[i] : 0;
    let end = i < endVals.length ? endVals[i] : dimSize;

    // Apply begin_mask: use 0 for forward, dimSize-1 for backward
    if (beginMask & (1 << i)) {
      start = stride > 0 ? 0 : dimSize - 1;
    }
    // Apply end_mask: use dimSize for forward, -dimSize-1 for backward
    if (endMask & (1 << i)) {
      end = stride > 0 ? dimSize : -(dimSize + 1);
    }

    // Normalize negative indices
    if (dimSize > 0) {
      if (start < 0) start += dimSize;
      if (end < 0) end += dimSize;
      start = Math.max(0, Math.min(start, dimSize));
      end = Math.max(0, Math.min(end, dimSize));
    }

    // shrink_axis_mask: slice exactly 1 element on this axis
    if (shrinkMask & (1 << i)) {
      shrinkAxes.push(i);
      sizes.push(1);
    } else {
      sizes.push(end - start);
    }

    starts.push(start);
  }

  // Check if all strides are 1
  const allStridesOne = strides.every(s => s === 1);

  if (allStridesOne) {
    if (shrinkAxes.length > 0) {
      const sliceVar = `${output}_slice`;
      emitter.line(`const ${sliceVar} = builder.slice(${input}, [${starts}], [${sizes}]);`);
      // Squeeze the shrunk axes — compute the output shape without those dims
      const outShape = sizes.filter((_, i) => !shrinkAxes.includes(i));
      emitter.line(`const ${output} = builder.reshape(${sliceVar}, [${outShape}]);`);
    } else {
      emitter.line(`const ${output} = builder.slice(${input}, [${starts}], [${sizes}]);`);
    }
  } else {
    if (shrinkAxes.length > 0) {
      const sliceVar = `${output}_slice`;
      emitter.line(`const ${sliceVar} = builder.slice(${input}, [${starts}], [${sizes}], { strides: [${strides}] });`);
      const outShape = sizes.filter((_, i) => !shrinkAxes.includes(i));
      emitter.line(`const ${output} = builder.reshape(${sliceVar}, [${outShape}]);`);
    } else {
      emitter.line(`const ${output} = builder.slice(${input}, [${starts}], [${sizes}], { strides: [${strides}] });`);
    }
  }
}

function emitPad(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // TFLite PAD/PADV2: inputs[1] is a constant [N, 2] int32 tensor
  // WebNN pad() requires (input, beginningPadding, endingPadding, options?)
  if (node.inputs.length >= 2 && emitter.isConstant(node.inputs[1])) {
    const rawData = emitter.constantRawData(node.inputs[1]);
    if (rawData) {
      const int32View = new Int32Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 4);
      const n = int32View.length / 2;
      const beginning: number[] = [];
      const ending: number[] = [];
      for (let i = 0; i < n; i++) {
        beginning.push(int32View[i * 2]);
        ending.push(int32View[i * 2 + 1]);
      }
      if (node.inputs.length >= 3 && node.inputs[2] !== '') {
        const constVal = emitter.ref(node.inputs[2]);
        emitter.line(`const ${output} = builder.pad(${input}, ${JSON.stringify(beginning)}, ${JSON.stringify(ending)}, { value: ${constVal} });`);
      } else {
        emitter.line(`const ${output} = builder.pad(${input}, ${JSON.stringify(beginning)}, ${JSON.stringify(ending)});`);
      }
      return;
    }
  }
  // Fallback: emit with empty padding arrays
  emitter.comment('WARNING: pad constant data not available');
  emitter.line(`const ${output} = builder.pad(${input}, [], []);`);
}

function emitMirrorPad(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  // TFLite MIRROR_PAD: inputs[1] is a constant [N, 2] int32 tensor
  if (node.inputs.length > 1 && emitter.isConstant(node.inputs[1])) {
    const rawData = emitter.constantRawData(node.inputs[1]);
    if (rawData) {
      const int32View = new Int32Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 4);
      const n = int32View.length / 2;
      const beginning: number[] = [];
      const ending: number[] = [];
      for (let i = 0; i < n; i++) {
        beginning.push(int32View[i * 2]);
        ending.push(int32View[i * 2 + 1]);
      }
      emitter.line(`const ${output} = builder.pad(${input}, ${JSON.stringify(beginning)}, ${JSON.stringify(ending)}, { mode: 'reflection' });`);
      return;
    }
  }
  // Fallback
  emitter.comment('WARNING: mirror_pad constant data not available');
  emitter.line(`const ${output} = builder.pad(${input}, [], [], { mode: 'reflection' });`);
}

function emitTile(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  if (node.inputs.length >= 2) {
    // Multiples is a constant int32 tensor
    const multiplesVals = emitter.constantIntValues(node.inputs[1]);
    if (multiplesVals) {
      emitter.line(`const ${output} = builder.tile(${input}, [${multiplesVals}]);`);
    } else {
      const multiples = emitter.ref(node.inputs[1]);
      emitter.line(`const ${output} = builder.tile(${input}, ${multiples});`);
    }
  } else {
    emitter.line(`const ${output} = ${input}; // TILE (missing multiples)`);
  }
}

function emitGather(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  let axis = (node.attributes.axis as number) ?? 0;
  // WebNN gather axis is unsigned long — normalize negative values
  if (axis < 0) {
    const inputShape = emitter.tensorShape(node.inputs[0]);
    if (inputShape) axis += inputShape.length;
  }
  emitter.line(`const ${output} = builder.gather(${input}, ${indices}, { axis: ${axis} });`);
}

function emitGatherND(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);

  emitter.line(`const ${output} = builder.gatherND(${input}, ${indices});`);
}

function emitScatterND(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const updates = emitter.ref(node.inputs[2]);
  const output = emitter.declare(node.outputs[0]);

  emitter.line(`const ${output} = builder.scatterND(${input}, ${indices}, ${updates});`);
}

function emitShape(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  emitter.line(`const ${output} = builder.shape(${input});`);
}

function emitPack(node: NodeIR, emitter: CodeEmitter): void {
  const inputs = node.inputs.map((n) => emitter.ref(n));
  const output = emitter.declare(node.outputs[0]);
  let axis = (node.attributes.axis as number) ?? 0;

  // WebNN concat axis is unsigned long — normalize negative values
  const inShape = emitter.tensorShape(node.inputs[0]);
  if (axis < 0 && inShape) {
    axis += inShape.length + 1; // pack inserts a dim, so rank+1
  }

  emitter.comment(`PACK along axis ${axis}`);
  // Pack = stack = reshape each input to add dim at axis, then concat
  const unsqueezed = inputs.map((inp, i) => {
    const uName = `${output}_u${i}`;
    if (inShape) {
      const newShape = [...inShape];
      newShape.splice(axis, 0, 1);
      emitter.line(`const ${uName} = builder.reshape(${inp}, [${newShape.join(', ')}]);`);
    } else {
      emitter.comment('WARNING: unknown shape for pack unsqueeze');
      emitter.line(`const ${uName} = ${inp};`);
    }
    return uName;
  });
  emitter.line(`const ${output} = builder.concat([${unsqueezed.join(', ')}], ${axis});`);
}

function emitUnpack(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  let axis = (node.attributes.axis as number) ?? 0;
  const num = (node.attributes.num as number) ?? node.outputs.length;

  // WebNN split axis is unsigned long — normalize negative values
  if (axis < 0) {
    const inputShape = emitter.tensorShape(node.inputs[0]);
    if (inputShape) axis += inputShape.length;
  }

  emitter.comment(`UNPACK along axis ${axis}`);
  const outShape = emitter.tensorShape(node.outputs[0]);
  for (let i = 0; i < node.outputs.length; i++) {
    const out = emitter.declare(node.outputs[i]);
    if (outShape) {
      emitter.line(`const ${out} = builder.reshape(builder.split(${input}, ${num}, { axis: ${axis} })[${i}], [${outShape.join(', ')}]);`);
    } else {
      // Fallback: split without squeeze
      emitter.line(`const ${out} = builder.split(${input}, ${num}, { axis: ${axis} })[${i}];`);
    }
  }
}

registerTfliteOp('RESHAPE', emitReshape);
registerTfliteOp('SQUEEZE', emitSqueeze);
registerTfliteOp('EXPAND_DIMS', emitExpandDims);
registerTfliteOp('TRANSPOSE', emitTranspose);
registerTfliteOp('CONCATENATION', emitConcatenation);
registerTfliteOp('SPLIT', emitSplit);
registerTfliteOp('SPLIT_V', emitSplitV);
registerTfliteOp('SLICE', emitSlice);
registerTfliteOp('STRIDED_SLICE', emitStridedSlice);
registerTfliteOp('PAD', emitPad);
registerTfliteOp('PADV2', emitPad);
registerTfliteOp('MIRROR_PAD', emitMirrorPad);
registerTfliteOp('TILE', emitTile);
registerTfliteOp('GATHER', emitGather);
registerTfliteOp('GATHER_ND', emitGatherND);
registerTfliteOp('SCATTER_ND', emitScatterND);
registerTfliteOp('SHAPE', emitShape);
registerTfliteOp('PACK', emitPack);
registerTfliteOp('UNPACK', emitUnpack);
