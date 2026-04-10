// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/
// Various shape/transform ops: Reshape, Transpose, Flatten, Squeeze, Unsqueeze,
// Concat, Split, Slice, Expand, Gather, Pad, Tile, Clip, Cast, Softmax, etc.

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

// reshape_op_builder.cc
// ORT resolves ONNX special values (-1 = infer, 0 = copy from input) via ReshapeHelper
// before passing uint32 shape to WebNN, which only accepts unsigned long values.
function resolveReshapeShape(
  targetShape: number[],
  inputShape: (number | string)[] | null,
  allowZero: number,
): number[] | null {
  const resolved = [...targetShape];
  const inputDims = inputShape && inputShape.every((d) => typeof d === 'number')
    ? (inputShape as number[])
    : null;

  // Replace 0 with corresponding input dim (when allowzero=0, the default)
  if (!allowZero && inputDims) {
    for (let i = 0; i < resolved.length; i++) {
      if (resolved[i] === 0 && i < inputDims.length) {
        resolved[i] = inputDims[i];
      }
    }
  }

  // Resolve -1: compute from total element count
  const inferIdx = resolved.indexOf(-1);
  if (inferIdx !== -1 && inputDims) {
    const totalInput = inputDims.reduce((a, b) => a * b, 1);
    const knownProduct = resolved.reduce((a, b, i) => (i === inferIdx ? a : a * b), 1);
    if (knownProduct > 0) {
      resolved[inferIdx] = totalInput / knownProduct;
    } else {
      return null; // cannot resolve
    }
  } else if (inferIdx !== -1) {
    return null; // cannot resolve without input shape
  }

  // Verify all values are non-negative integers
  if (resolved.some((d) => d < 0 || !Number.isInteger(d))) return null;
  return resolved;
}

function emitReshape(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const allowZero = (node.attributes.allowzero as number) ?? 0;
  // Shape input must be a constant initializer (per ORT validation)
  if (emitter.isConstant(node.inputs[1])) {
    const shapeValues = emitter.constantIntValues(node.inputs[1]);
    if (shapeValues) {
      const inputShape = emitter.tensorShape(node.inputs[0]);
      const resolved = resolveReshapeShape(shapeValues, inputShape, allowZero);
      if (resolved) {
        emitter.line(`const ${output} = builder.reshape(${input}, [${resolved.join(', ')}]);`);
      } else {
        // Fallback: if output shape is known and fully static, use it directly
        const outputShape = emitter.tensorShape(node.outputs[0]);
        if (outputShape && outputShape.every((d) => typeof d === 'number' && d >= 0)) {
          emitter.line(`const ${output} = builder.reshape(${input}, [${outputShape.join(', ')}]);`);
        } else {
          emitter.comment(`WARNING: Could not resolve reshape shape [${shapeValues.join(', ')}] — input shape unknown`);
          emitter.line(`const ${output} = builder.reshape(${input}, [${shapeValues.join(', ')}]);`);
        }
      }
    } else {
      // Fallback: reference the constant operand directly
      const shape = emitter.ref(node.inputs[1]);
      emitter.line(`const ${output} = builder.reshape(${input}, ${shape});`);
    }
  } else {
    // Dynamic shape input (not a constant initializer) — try to use the statically-resolved
    // output shape from propagateShapes, which folds Shape/Slice/Concat chains at parse time.
    const outputShape = emitter.tensorShape(node.outputs[0]);
    if (outputShape && outputShape.length > 0 && outputShape.every((d): d is number => typeof d === 'number' && d >= 0)) {
      emitter.line(`const ${output} = builder.reshape(${input}, [${outputShape.join(', ')}]);`);
    } else {
      emitter.comment(`WARNING: Dynamic reshape with unresolved shape — marking dead`);
      emitter.line(`const ${output} = undefined; // dead`);
      emitter.markDead(node.outputs[0]);
    }
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

  const shape = emitter.tensorShape(node.inputs[0]);
  if (shape && shape.length > 0) {
    const rank = shape.length;
    const effectiveAxis = axis < 0 ? axis + rank : axis;
    const preDims = shape.slice(0, effectiveAxis);
    const postDims = shape.slice(effectiveAxis);
    // Compute products — treat dynamic (string) dims as 1 for inference
    const preProduct = preDims.reduce<number>(
      (a, d) => a * (typeof d === 'number' ? d : 1), 1,
    );
    const postProduct = postDims.reduce<number>(
      (a, d) => a * (typeof d === 'number' ? d : 1), 1,
    );
    emitter.line(`const ${output} = builder.reshape(${input}, [${preProduct}, ${postProduct}]);`);
  } else {
    emitter.comment(`WARNING: Flatten without shape info, axis=${axis}`);
    emitter.line(`const ${output} = builder.reshape(${input}, []); // TODO: provide shape for flatten`);
  }
}

// squeeze_unsqueeze_op_builder.cc
// ORT reads axes from attribute (opset <13) or constant initializer (opset 13+),
// resolves negative indices, then uses reshape to implement squeeze/unsqueeze.
// WebNN has no squeeze/unsqueeze ops — ORT decomposes both to reshape.
function emitSqueeze(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  let axes = node.attributes.axes as number[] | undefined;
  // Opset 13+: axes as second input constant
  if (!axes && node.inputs.length > 1 && node.inputs[1] !== '' && emitter.isConstant(node.inputs[1])) {
    axes = emitter.constantIntValues(node.inputs[1]) ?? undefined;
  }
  // Use output shape from graph if available
  const outputShape = emitter.tensorShape(node.outputs[0]);
  if (outputShape && outputShape.every((d) => typeof d === 'number' && d >= 0)) {
    emitter.line(`const ${output} = builder.reshape(${input}, [${outputShape.join(', ')}]);`);
    return;
  }
  // Compute output shape by removing axes from input shape
  const inputShape = emitter.tensorShape(node.inputs[0]);
  if (inputShape && inputShape.every((d) => typeof d === 'number') && axes) {
    const rank = inputShape.length;
    const resolved = new Set(axes.map((a) => (a < 0 ? a + rank : a)));
    const newShape = (inputShape as number[]).filter((_, i) => !resolved.has(i));
    emitter.line(`const ${output} = builder.reshape(${input}, [${newShape.join(', ')}]);`);
  } else if (inputShape && inputShape.every((d) => typeof d === 'number') && !axes) {
    // No axes specified: remove all dimensions of size 1
    const newShape = (inputShape as number[]).filter((d) => d !== 1);
    emitter.line(`const ${output} = builder.reshape(${input}, [${newShape.join(', ')}]);`);
  } else {
    emitter.comment(`WARNING: Cannot resolve squeeze — missing shape info`);
    emitter.line(`const ${output} = ${input}; // squeeze fallback`);
  }
}

function emitUnsqueeze(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  let axes = node.attributes.axes as number[] | undefined;
  // Opset 13+: axes as second input constant
  if (!axes && node.inputs.length > 1 && node.inputs[1] !== '' && emitter.isConstant(node.inputs[1])) {
    axes = emitter.constantIntValues(node.inputs[1]) ?? undefined;
  }
  // Use output shape from graph if available
  const outputShape = emitter.tensorShape(node.outputs[0]);
  if (outputShape && outputShape.every((d) => typeof d === 'number' && d >= 0)) {
    emitter.line(`const ${output} = builder.reshape(${input}, [${outputShape.join(', ')}]);`);
    return;
  }
  // Compute output shape by inserting 1s at the specified axes
  const inputShape = emitter.tensorShape(node.inputs[0]);
  if (inputShape && inputShape.every((d) => typeof d === 'number') && axes) {
    const expandedRank = inputShape.length + axes.length;
    const resolved = axes.map((a) => (a < 0 ? a + expandedRank : a)).sort((a, b) => a - b);
    const newShape: number[] = [];
    let srcIdx = 0;
    for (let i = 0; i < expandedRank; i++) {
      if (resolved.includes(i)) {
        newShape.push(1);
      } else {
        newShape.push(inputShape[srcIdx] as number);
        srcIdx++;
      }
    }
    emitter.line(`const ${output} = builder.reshape(${input}, [${newShape.join(', ')}]);`);
  } else {
    emitter.comment(`WARNING: Cannot resolve unsqueeze — missing shape or axes info`);
    emitter.line(`const ${output} = ${input}; // unsqueeze fallback`);
  }
}

// concat_op_builder.cc
function emitConcat(node: NodeIR, emitter: CodeEmitter): void {
  // Propagate dead state: if any input is dead, skip
  if (node.inputs.some((name) => emitter.isDead(name))) {
    emitter.comment(`Concat skipped: has dead input(s)`);
    emitter.markDead(node.outputs[0]);
    return;
  }

  const inputs = node.inputs.map((name) => emitter.ref(name));
  const output = emitter.declare(node.outputs[0]);
  let axis = (node.attributes.axis as number) ?? 0;
  // WebNN axis is unsigned long — normalize negative values
  if (axis < 0) {
    const inputShape = emitter.tensorShape(node.inputs[0]);
    if (inputShape) axis = axis + inputShape.length;
  }
  emitter.line(`const ${output} = builder.concat([${inputs.join(', ')}], ${axis});`);
}

// split_op_builder.cc
// ORT reads split sizes from attribute (opset <13) or constant initializer (opset 13+).
function emitSplit(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  let axis = (node.attributes.axis as number) ?? 0;
  // WebNN axis is unsigned long — normalize negative values
  if (axis < 0) {
    const inputShape = emitter.tensorShape(node.inputs[0]);
    if (inputShape) axis = axis + inputShape.length;
  }
  const numOutputs = node.outputs.length;

  // Try attribute first, then second input constant (opset 13+)
  let split = node.attributes.split as number[] | undefined;
  if (!split && node.inputs.length > 1 && node.inputs[1] !== '' && emitter.isConstant(node.inputs[1])) {
    split = emitter.constantIntValues(node.inputs[1]) ?? undefined;
  }

  const splitVar = emitter.declare(`${node.outputs[0]}_splits`);
  if (split) {
    emitter.line(`const ${splitVar} = builder.split(${input}, [${split.join(', ')}], { axis: ${axis} });`);
  } else {
    emitter.line(`const ${splitVar} = builder.split(${input}, ${numOutputs}, { axis: ${axis} });`);
  }
  for (let i = 0; i < numOutputs; i++) {
    const out = emitter.declare(node.outputs[i]);
    emitter.line(`const ${out} = ${splitVar}[${i}];`);
  }
}

// slice_op_builder.cc
// ORT reads starts/ends/axes/steps from constant initializers and resolves them to
// JavaScript arrays for WebNN's builder.slice(input, starts, sizes, { strides }).
function emitSlice(node: NodeIR, emitter: CodeEmitter): void {
  // Propagate dead state: if data input is dead, skip
  if (emitter.isDead(node.inputs[0])) {
    emitter.comment(`Slice skipped: input ${node.inputs[0]} is dead`);
    emitter.markDead(node.outputs[0]);
    return;
  }

  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const inputShape = emitter.tensorShape(node.inputs[0]);
  const rank = inputShape ? inputShape.length : 0;

  // Extract constant integer values for starts, ends, axes, steps
  const startsRaw = emitter.isConstant(node.inputs[1]) ? emitter.constantIntValues(node.inputs[1]) : null;
  const endsRaw = emitter.isConstant(node.inputs[2]) ? emitter.constantIntValues(node.inputs[2]) : null;
  const axesRaw = node.inputs.length > 3 && node.inputs[3] !== '' && emitter.isConstant(node.inputs[3])
    ? emitter.constantIntValues(node.inputs[3]) : null;
  const stepsRaw = node.inputs.length > 4 && node.inputs[4] !== '' && emitter.isConstant(node.inputs[4])
    ? emitter.constantIntValues(node.inputs[4]) : null;

  if (startsRaw && endsRaw && inputShape && inputShape.every((d) => typeof d === 'number')) {
    const shape = inputShape as number[];
    const n = startsRaw.length;

    // Determine axes (default: [0, 1, ..., n-1])
    const axes = axesRaw ? axesRaw.map((a) => (a < 0 ? a + rank : a)) : Array.from({ length: n }, (_, i) => i);
    const steps = stepsRaw ?? new Array(n).fill(1);

    // Build per-axis starts/sizes/strides arrays (full rank)
    const resolvedStarts = new Array(rank).fill(0);
    const resolvedSizes = shape.slice();
    const resolvedStrides = new Array(rank).fill(1);

    for (let i = 0; i < n; i++) {
      const ax = axes[i];
      const dimSize = shape[ax];
      const step = steps[i];
      // Clamp start/end per ONNX spec
      let s = startsRaw[i];
      let e = endsRaw[i];
      if (s < 0) s = Math.max(0, s + dimSize);
      if (e < 0) e = Math.max(0, e + dimSize);
      s = Math.min(s, dimSize);
      e = Math.min(e, dimSize);
      // Compute size = ceil((end - start) / step)
      const size = Math.max(0, Math.ceil((e - s) / step));
      resolvedStarts[ax] = s;
      resolvedSizes[ax] = size;
      resolvedStrides[ax] = step;
    }

    const hasStrides = resolvedStrides.some((s) => s !== 1);
    const stridesOpt = hasStrides ? `, { strides: [${resolvedStrides.join(', ')}] }` : '';
    emitter.line(`const ${output} = builder.slice(${input}, [${resolvedStarts.join(', ')}], [${resolvedSizes.join(', ')}]${stridesOpt});`);
  } else {
    // Fallback: try to use the output shape to derive sizes
    const outputShape = emitter.tensorShape(node.outputs[0]);
    if (startsRaw && outputShape && outputShape.every((d) => typeof d === 'number')) {
      // We have starts and the output shape is fully static — use output shape as sizes
      const sizes = outputShape as number[];
      const resolvedStarts = new Array(rank).fill(0);
      const resolvedSizes = inputShape && inputShape.every((d) => typeof d === 'number')
        ? (inputShape as number[]).slice() : sizes.slice();
      const resolvedStrides = new Array(rank).fill(1);
      const axes = axesRaw ? axesRaw.map((a) => (a < 0 ? a + rank : a)) : Array.from({ length: startsRaw.length }, (_, i) => i);
      const steps = stepsRaw ?? new Array(startsRaw.length).fill(1);
      for (let i = 0; i < startsRaw.length; i++) {
        const ax = axes[i];
        const dimSize = resolvedSizes[ax] || sizes[ax] || 0;
        let s = startsRaw[i];
        if (s < 0) s = Math.max(0, s + dimSize);
        resolvedStarts[ax] = s;
        resolvedSizes[ax] = sizes[ax];
        resolvedStrides[ax] = steps[i];
      }
      const hasStrides = resolvedStrides.some((s) => s !== 1);
      const stridesOpt = hasStrides ? `, { strides: [${resolvedStrides.join(', ')}] }` : '';
      emitter.comment(`Slice: used output shape as sizes (ends not resolved as constants)`);
      emitter.line(`const ${output} = builder.slice(${input}, [${resolvedStarts.join(', ')}], [${resolvedSizes.join(', ')}]${stridesOpt});`);
    } else {
      // Cannot resolve — WebNN builder.slice requires plain JS arrays, not MLOperand refs
      emitter.comment(`ERROR: Slice inputs (starts/ends) could not be resolved as constant arrays.`);
      emitter.comment(`WebNN builder.slice() requires plain JavaScript arrays for starts and sizes.`);
      emitter.line(`const ${output} = (() => { throw new Error('Slice: starts/ends not resolvable as constants — ensure all free dimensions are specified'); })();`);
    }
  }
}

// expand_op_builder.cc
// ORT reads shape from constant initializer, converts int64→uint32, passes JS array.
function emitExpand(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  if (emitter.isConstant(node.inputs[1])) {
    const shapeValues = emitter.constantIntValues(node.inputs[1]);
    if (shapeValues) {
      emitter.line(`const ${output} = builder.expand(${input}, [${shapeValues.join(', ')}]);`);
      return;
    }
  }
  // Fallback: try output shape
  const outputShape = emitter.tensorShape(node.outputs[0]);
  if (outputShape && outputShape.every((d) => typeof d === 'number')) {
    emitter.line(`const ${output} = builder.expand(${input}, [${outputShape.join(', ')}]);`);
  } else {
    emitter.comment(`WARNING: Expand shape not resolved as constant`);
    const shape = emitter.ref(node.inputs[1]);
    emitter.line(`const ${output} = builder.expand(${input}, ${shape});`);
  }
}

// gather_op_builder.cc
function emitGather(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const indices = emitter.ref(node.inputs[1]);
  const output = emitter.declare(node.outputs[0]);
  let axis = (node.attributes.axis as number) ?? 0;
  // WebNN axis is unsigned long — normalize negative values
  if (axis < 0) {
    const inputShape = emitter.tensorShape(node.inputs[0]);
    if (inputShape) axis = axis + inputShape.length;
  }
  emitter.line(`const ${output} = builder.gather(${input}, ${indices}, { axis: ${axis} });`);
}

// pad_op_builder.cc
function emitPad(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  const mode = (node.attributes.mode as string) ?? 'constant';

  // ONNX pads: 1D int64 tensor [begin_0, begin_1, ..., end_0, end_1, ...]
  // WebNN pad() requires (input, beginningPadding, endingPadding, options?)
  if (emitter.isConstant(node.inputs[1])) {
    const rawData = emitter.constantRawData(node.inputs[1]);
    const dataType = emitter.constantDataType(node.inputs[1]);
    if (rawData) {
      let values: number[];
      if (dataType === 'int64') {
        const view = new BigInt64Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 8);
        values = Array.from(view, (v) => Number(v));
      } else {
        const view = new Int32Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 4);
        values = Array.from(view);
      }
      const half = values.length / 2;
      const beginning = values.slice(0, half);
      const ending = values.slice(half);

      const opts: string[] = [];
      if (mode !== 'constant') {
        opts.push(`mode: '${mode}'`);
      }
      if (node.inputs.length > 2 && node.inputs[2] !== '') {
        opts.push(`value: ${emitter.ref(node.inputs[2])}`);
      }
      const optStr = opts.length > 0 ? `, { ${opts.join(', ')} }` : '';
      emitter.line(`const ${output} = builder.pad(${input}, ${JSON.stringify(beginning)}, ${JSON.stringify(ending)}${optStr});`);
      return;
    }
  }
  // Fallback
  emitter.comment('WARNING: pad constant data not available');
  emitter.line(`const ${output} = builder.pad(${input}, [], []);`);
}

// tile_op_builder.cc
// ORT reads repetitions from constant initializer, converts int64→uint32, passes JS array.
function emitTile(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);
  if (emitter.isConstant(node.inputs[1])) {
    const repsValues = emitter.constantIntValues(node.inputs[1]);
    if (repsValues) {
      emitter.line(`const ${output} = builder.tile(${input}, [${repsValues.join(', ')}]);`);
      return;
    }
  }
  emitter.comment(`WARNING: Tile repetitions not resolved as constant`);
  const repeats = emitter.ref(node.inputs[1]);
  emitter.line(`const ${output} = builder.tile(${input}, ${repeats});`);
}

// clip_op_builder.cc → clamp
// ORT extracts scalar float values for min/max via GetClipMinMax.
// WebNN clamp expects MLNumber (scalar), not MLOperand.

/** Decode a scalar float16 stored as 2 little-endian bytes into a JS number. */
function decodeFloat16(raw: Uint8Array): number {
  const u16 = raw[0] | (raw[1] << 8);
  const sign = (u16 >> 15) ? -1 : 1;
  const exp = (u16 >> 10) & 0x1f;
  const mantissa = u16 & 0x3ff;
  if (exp === 0x1f) return mantissa ? NaN : sign * Infinity;
  if (exp === 0) return sign * Math.pow(2, -14) * (mantissa / 1024);
  return sign * Math.pow(2, exp - 15) * (1 + mantissa / 1024);
}

/** Extract a scalar number from a constant tensor (supports float32 and float16). */
function extractScalarClampValue(emitter: CodeEmitter, tensorName: string): number | null {
  const raw = emitter.constantRawData(tensorName);
  if (!raw) return null;
  if (raw.byteLength === 4) {
    const aligned = new ArrayBuffer(4);
    new Uint8Array(aligned).set(raw);
    return new Float32Array(aligned)[0];
  }
  if (raw.byteLength === 2) {
    return decodeFloat16(raw);
  }
  return null;
}

function emitClip(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  const hasMin = node.inputs.length > 1 && node.inputs[1] !== '';
  const hasMax = node.inputs.length > 2 && node.inputs[2] !== '';

  const opts: string[] = [];
  if (hasMin) {
    if (emitter.isConstant(node.inputs[1])) {
      const val = extractScalarClampValue(emitter, node.inputs[1]);
      if (val !== null) {
        opts.push(`minValue: ${val}`);
      } else {
        opts.push(`minValue: ${emitter.ref(node.inputs[1])}`);
      }
    } else {
      opts.push(`minValue: ${emitter.ref(node.inputs[1])}`);
    }
  }
  if (hasMax) {
    if (emitter.isConstant(node.inputs[2])) {
      const val = extractScalarClampValue(emitter, node.inputs[2]);
      if (val !== null) {
        opts.push(`maxValue: ${val}`);
      } else {
        opts.push(`maxValue: ${emitter.ref(node.inputs[2])}`);
      }
    } else {
      opts.push(`maxValue: ${emitter.ref(node.inputs[2])}`);
    }
  }

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
  let axis = (node.attributes.axis as number) ?? -1;
  // WebNN axis is unsigned long — normalize negative values
  if (axis < 0) {
    const inputShape = emitter.tensorShape(node.inputs[0]);
    axis = inputShape ? axis + inputShape.length : 1;
  }
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
// WebNN doesn't have a Shape op. ORT decomposes it to constant + slice.
// Creates a constant tensor from the known input shape, then slices per start/end attrs.
function emitShape(node: NodeIR, emitter: CodeEmitter): void {
  const inputShape = emitter.tensorShape(node.inputs[0]);

  if (!inputShape || inputShape.length === 0) {
    emitter.comment(`Shape op skipped: input shape unknown for ${node.inputs[0]}`);
    emitter.markDead(node.outputs[0]);
    return;
  }

  // Resolve shape: use numeric values directly, treat dynamic dims as 1
  // (dynamic dims are symbolic names from the model — at build time we use 1 as default)
  const output = emitter.declare(node.outputs[0]);
  const dims = inputShape.map((d) => (typeof d === 'number' ? d : 1));
  const rank = dims.length;

  // Handle start/end attributes (opset 15+)
  let start = (node.attributes.start as number) ?? 0;
  let end = (node.attributes.end as number) ?? rank;
  // Resolve negatives and clamp
  start = Math.max(0, Math.min(rank, start + (start < 0 ? rank : 0)));
  end = Math.max(start, Math.min(rank, end + (end < 0 ? rank : 0)));

  const slicedDims = dims.slice(start, end);
  // Emit as a constant int64 tensor (matching ORT behavior)
  emitter.line(`const ${output} = builder.constant({ dataType: 'int64', shape: [${slicedDims.length}] }, new BigInt64Array([${slicedDims.map((d) => d + 'n').join(', ')}]));`);
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
