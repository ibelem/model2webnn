---
applyTo: "src/operators/**"
description: "Guidelines for implementing WebNN operator builders. Use when creating or modifying operator mapping code in src/operators/."
---

# Operator Implementation Guide

## Source of Truth

Every operator implementation MUST be ported from the corresponding C++ file in:
```
reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/
```

## File Mapping

One TypeScript file per ORT builder file. Name matches the ORT file:

| ORT File | Our File | ONNX Ops |
|---------|----------|----------|
| `activation_op_builder.cc` | `src/operators/onnx/activation.ts` | Elu, Gelu, HardSigmoid, HardSwish, LeakyRelu, Relu, Sigmoid, Softplus, Softsign, Tanh |
| `binary_op_builder.cc` | `src/operators/onnx/binary.ts` | Add, Sub, Mul, Div, Pow, PRelu |
| `conv_op_builder.cc` | `src/operators/onnx/conv.ts` | Conv, ConvInteger, ConvTranspose |
| `gemm_op_builder.cc` | `src/operators/onnx/gemm.ts` | Gemm, MatMul, MatMulInteger |
| `gqa_op_builder.cc` | `src/operators/onnx/gqa.ts` | GroupQueryAttention |
| `mha_op_builder.cc` | `src/operators/onnx/mha.ts` | MultiHeadAttention |
| ... | ... | ... |

## Implementation Pattern

Each op builder file exports a function that generates WebNN code for one or more ONNX ops.

### 1:1 Ops (Direct WebNN Mapping)

For simple ops like activations:

```typescript
// src/operators/onnx/activation.ts
import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../../codegen/emitter.js';

export function emitActivation(node: NodeIR, emitter: CodeEmitter): void {
  const input = emitter.ref(node.inputs[0]);
  const output = emitter.declare(node.outputs[0]);

  switch (node.opType) {
    case 'Relu':
      emitter.line(`const ${output} = builder.relu(${input});`);
      break;
    case 'Sigmoid':
      emitter.line(`const ${output} = builder.sigmoid(${input});`);
      break;
    case 'Elu': {
      const alpha = node.attributes.alpha ?? 1.0;
      emitter.line(`const ${output} = builder.elu(${input}, { alpha: ${alpha} });`);
      break;
    }
    // ... follow activation_op_builder.cc for all cases
  }
}
```

### Composite Ops (Decomposed)

For ops like GQA that decompose into multiple WebNN calls, follow the ORT
builder step-by-step. Include comments referencing the C++ source:

```typescript
// src/operators/onnx/gqa.ts
// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/gqa_op_builder.cc

export function emitGQA(node: NodeIR, emitter: CodeEmitter): void {
  // Step 1: Split packed QKV (if key/value not provided)
  // See gqa_op_builder.cc lines ~100-130
  // ...

  // Step 2: Apply rotary embedding (if do_rotary attribute is true)
  // See gqa_op_builder.cc lines ~130-200
  // ...
}
```

## Rules

1. **Follow ORT builder logic exactly** — preserve attribute defaults, edge cases, optional inputs
2. **No layout conversion** — pass weights through as-is
3. **Use node.attributes** — read ONNX attributes using same names/defaults as ORT
4. **Handle optional inputs** — check `node.inputs[i]` exists before accessing
5. **Comment deviations** — if you must deviate from ORT, add a `// DEVIATION:` comment with rationale
6. **Register in registry.ts** — every op builder must be registered in the central dispatch

## WebNN API Reference

Check `reference/webnn-spec/webnn.idl.txt` for:
- Exact method signatures on `MLGraphBuilder`
- Option dictionary types and their fields
- Valid data type strings

## Common Attribute Patterns

```typescript
// Integer attribute with default
const axis = node.attributes.axis ?? 0;

// Integer array attribute
const strides = node.attributes.strides ?? [1, 1];
const pads = node.attributes.pads ?? [0, 0, 0, 0];

// Float attribute
const alpha = node.attributes.alpha ?? 1.0;

// String attribute
const direction = node.attributes.direction ?? 'forward';

// Optional input (may be empty string or undefined)
const hasBias = node.inputs.length > 2 && node.inputs[2] !== '';
```
