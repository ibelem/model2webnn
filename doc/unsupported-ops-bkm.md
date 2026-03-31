# BKM: Handling WebNN-Unsupported Ops (TopK, Range, Mod, etc.)

## Summary

WebNN has no equivalents for several ONNX/TFLite operators such as `TopK`, `Range`, and `Mod`. ORT's WebNN backend also does not support these ops. The code generator uses a **dead-propagation** strategy: unsupported op outputs are marked "dead", all downstream ops that depend on dead tensors are automatically skipped, and the graph exports the last live "frontier" tensors instead of the original model outputs.

## Problem

Models like YOLO use post-processing ops (NMS-style TopK filtering, index arithmetic with Range/Mod) that have no WebNN mapping. When these ops appeared in `yolo26n_int8.onnx`, the code generator originally emitted `undefined` for their outputs but still generated downstream ops like `builder.reshape(undefined, ...)`, causing:

```
Error: Failed to execute 'reshape' on 'MLGraphBuilder': parameter 1 is not of type 'MLOperand'.
```

After fixing that to skip downstream ops, all graph outputs ended up dead, causing:

```
Error: Failed to execute 'build' on 'MLGraphBuilder': At least one output needs to be provided.
```

And after fixing the build, the HTML harness still referenced the original (now-dead) outputs, causing:

```
Error: Failed to execute 'dispatch' on 'MLContext': Invalid outputs: The number (1) of MLTensor(s) doesn't match the expectation (5).
```

### Unsupported ops confirmed

| Op | In WebNN spec? | In ORT WebNN backend? |
|----|---------------|----------------------|
| TopK | No | No |
| Range | No | No |
| Mod | No | No |

Verified against `reference/webnn-spec/webnn.idl.txt` and `reference/microsoft/onnxruntime/core/providers/webnn/builders/op_builder_factory.cc`.

## Solution

Three coordinated changes in `src/codegen/javascript.ts` and `src/codegen/html.ts`:

### 1. Mark unsupported op outputs as dead

When an op has no registered emitter, its outputs are marked dead via `emitter.markDead()` instead of just assigning `undefined`:

```typescript
// In the node dispatch loop (both generateJavaScript and generateJavaScriptFixed):
const opEmitter = getEmitter(graph.format, node.opType);
if (opEmitter) {
  emit(`// ${node.opType}`);
  opEmitter(node, emitterImpl);
} else {
  emit(`// UNSUPPORTED: ${node.opType} — no WebNN equivalent`);
  for (const out of node.outputs) {
    if (out !== '') {
      emitterImpl.markDead(out);
      const varName = getOrDeclare(out);
      emit(`const ${varName} = undefined; // unsupported op`);
    }
  }
}
```

### 2. Propagate dead state through all downstream ops

Before dispatching any node to its op emitter, the loop checks if any input is dead. If so, all outputs are marked dead and the node is skipped entirely — no WebNN builder calls are emitted:

```typescript
for (const node of graph.nodes) {
  const hasDead = node.inputs.some(
    (name) => name !== '' && emitterImpl.isDead(name),
  );
  if (hasDead) {
    emit(`// SKIPPED: ${node.opType} — depends on unsupported op output`);
    for (const out of node.outputs) {
      if (out !== '') {
        emitterImpl.markDead(out);
        const varName = getOrDeclare(out);
        emit(`const ${varName} = undefined; // dead — upstream unsupported`);
      }
    }
    continue;
  }
  // ... normal dispatch
}
```

This is applied **at the dispatch level** in both `generateJavaScript` and `generateJavaScriptFixed`, so every op automatically benefits without needing per-op `isDead()` checks.

### 3. Export frontier tensors when all outputs are dead

When all original graph outputs are dead, `findFrontierTensors()` identifies the last live intermediate tensors before the dead zone — these are live tensors consumed by at least one dead node, excluding constants and graph inputs:

```typescript
function findFrontierTensors(graph: GraphIR, emitter: CodeEmitter): string[] {
  const frontier = new Set<string>();
  for (const node of graph.nodes) {
    const anyOutputDead = node.outputs.some((o) => o !== '' && emitter.isDead(o));
    if (!anyOutputDead) continue;
    for (const inp of node.inputs) {
      if (inp === '' || emitter.isDead(inp)) continue;
      if (constantNames.has(inp) || graphInputNames.has(inp)) continue;
      frontier.add(inp);
    }
  }
  return [...frontier];
}
```

The build section uses this fallback:

```typescript
const liveOutputs = graph.outputs.filter((o) => !emitterImpl.isDead(o.name));
if (liveOutputs.length > 0) {
  // Normal case — use original outputs
} else {
  // All outputs dead — export frontier tensors
  const frontier = findFrontierTensors(graph, emitterImpl);
  for (const name of frontier) {
    emit(`namedOutputs['${name}'] = ${getOrDeclare(name)};`);
  }
}
```

### 4. Sync HTML harness with effective outputs

`computeEffectiveOutputs()` replicates the dead-propagation logic (without codegen) to determine which tensors the generated `buildGraph()` will actually return. The HTML generator uses this to build `OUTPUT_INFO`:

```typescript
const effectiveOutputs = computeEffectiveOutputs(graph);
// ... build OUTPUT_INFO from effectiveOutputs instead of graph.outputs
```

This ensures `context.dispatch(graph, inputs, outputs)` sees matching output tensor counts.

## Example: yolo26n_int8

The model has 1080 ops. The last ~30 ops are NMS-style post-processing using TopK/Range/Mod.

**Before:** 1 output (`logits`) — dead due to TopK dependency.

**After:** 5 frontier outputs:

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `/model.23/Split_output_0` | [1, 8400, 4] | float32 | Detection boxes |
| `/model.23/Split_output_1` | [1, 8400, 1] | float32 | Object confidence scores |
| `/model.23/Split_output_2` | [1, 8400, 51] | float32 | Keypoint coordinates |
| `/model.23/ReduceMax_output_0` | [1, 8400] | float32 | Max class scores |
| `/model.23/Cast_6_output_0` | [] | int64 | Shape metadata (scalar) |

The user can perform NMS post-processing in JavaScript on these raw detection outputs.

## Generated code pattern

```javascript
// ... 1076 supported ops emitted normally ...

// UNSUPPORTED: TopK — no WebNN equivalent
const _model_23_TopK_output_0 = undefined; // unsupported op
const _model_23_TopK_output_1 = undefined; // unsupported op
// SKIPPED: Unsqueeze — depends on unsupported op output
const _model_23_Expand_6_output_0 = undefined; // dead — upstream unsupported
// UNSUPPORTED: Range — no WebNN equivalent
const _model_23_Range_6_output_0 = undefined; // unsupported op
// ... 24 more skipped ops ...

// Build graph
const namedOutputs = {};
// NOTE: All original outputs depend on unsupported ops (e.g. TopK, Range, Mod).
// Exporting the last computed tensors before the unsupported section.
namedOutputs['/model.23/ReduceMax_output_0'] = _model_23_ReduceMax_output_0;
namedOutputs['/model.23/Split_output_1'] = _model_23_Split_output_1;
namedOutputs['/model.23/Split_output_0'] = _model_23_Split_output_0;
namedOutputs['/model.23/Split_output_2'] = _model_23_Split_output_2;
namedOutputs['/model.23/Cast_6_output_0'] = _model_23_Cast_6_output_0;
return await builder.build(namedOutputs);
```

## Files changed

| File | Change |
|------|--------|
| `src/codegen/javascript.ts` | Dead propagation in dispatch loop, frontier fallback in build section, `findFrontierTensors()`, `computeEffectiveOutputs()` |
| `src/codegen/html.ts` | Use `computeEffectiveOutputs()` for `OUTPUT_INFO` and output table |
