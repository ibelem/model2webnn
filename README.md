# model2webnn

Convert `.onnx` and `.tflite` models into ready-to-run [WebNN API](https://www.w3.org/TR/webnn/) JavaScript.

[**Try it online**](https://ibelem.github.io/model2webnn) · [WebNN spec](https://www.w3.org/TR/webnn/) · [WebNN Netron](https://ibelem.github.io/netron/)

## Overview

model2webnn parses `.onnx` and `.tflite` model files and generates self-contained JavaScript that uses the [WebNN `MLGraphBuilder`](https://www.w3.org/TR/webnn/#api-mlgraphbuilder) API. All processing is client-side — models never leave the browser.

**Inputs:** ONNX (`.onnx`), TFLite (`.tflite`)  
**Outputs:** `.js`, `.ts`, `.html`, `.weights` (WebNN Graph Weights binary), `.manifest.json`

---

## Getting started

### Web UI

1. Open [ibelem.github.io/model2webnn](https://ibelem.github.io/model2webnn)
2. Upload a model or paste a URL (HuggingFace links supported)
3. Code generates instantly in the Monaco editor
4. Set free dimension overrides to regenerate with fixed shapes
5. Download individual files or the full `.zip` bundle from the header

**Direct URL fetch** — append `?url=` to auto-load a model on open:

```
https://ibelem.github.io/model2webnn?url=https://huggingface.co/webnn/mobilenet-v2/resolve/main/onnx/model_fp16.onnx
```

**URL with dimension overrides** — add `&dim=name:value` to pre-fill symbolic dimensions and auto-convert:

```
https://ibelem.github.io/model2webnn?url=https://example.com/text_model.onnx&dim=batch_size:1&dim=sequence_length:77
```

Dimension values sync back to the URL as you change them in the UI.

### CLI

```bash
npx tsx src/cli.ts model.onnx -o dist/
npx tsx src/cli.ts model.onnx -o dist/ -f ts
npx tsx src/cli.ts model.onnx -o dist/ -d batch_size=1 -d seq_len=128
npx tsx src/cli.ts --list-ops
```

| Flag | Description |
|------|-------------|
| `-o, --output <dir>` | Output directory (default: `.`) |
| `-f, --format <fmt>` | `js`, `ts`, or `html` (default: `js`) |
| `-d, --free-dim <n=v>` | Override a symbolic dimension; repeatable |
| `--list-ops` | Print all supported operations and exit |

### Library API

```typescript
import { convert } from 'model2webnn';

const buffer = new Uint8Array(await file.arrayBuffer());
const result = await convert(buffer, {
  format: 'javascript',
  freeDimensionOverrides: { batch_size: 1 },
});

result.code;               // .js source — buildGraph(context, weights)
result.weights;            // Uint8Array — WebNN Graph WeighTs (WGWT) binary
result.manifest;           // tensor name → { dataType, shape, byteOffset, byteLength }
result.html;               // self-contained test page
result.coverage;           // { totalOps, supportedOps, unsupportedOpTypes, … }
result.unresolvedFreeDims; // symbolic dimension names not yet overridden
```

---

## Output files

| File | Description |
|------|-------------|
| `model.js` | `async buildGraph(context, weights)` — pure `MLGraphBuilder` calls, no framework |
| `model.weights` | WebNN Graph WeighTs (WGWT) v1 binary — raw tensor data with 8-byte header |
| `model.manifest.json` | Tensor index: names, shapes, data types, byte offsets |
| `model.html` | Runnable test harness with device selector and result viewer |

### WebNN Graph WeighTs (WGWT) weight format

```
Bytes 0–3   WebNN Graph WeighTs / WGWT  magic (ASCII)
Bytes 4–7   1       version (little-endian u32)
Bytes 8+    raw tensor data (concatenated, no padding)
```

`model.manifest.json` maps each tensor name to its slice:

```json
{
  "format": "wg-weights-manifest",
  "version": 1,
  "endianness": "little",
  "tensors": {
    "conv1.weight": {
      "dataType": "float32",
      "shape": [32, 3, 3, 3],
      "byteOffset": 8,
      "byteLength": 3456
    }
  }
}
```

---

## Operator coverage

Operator implementations follow the [ORT WebNN Execution Provider](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/webnn) 1:1 — same attribute defaults and edge-case handling.

### ONNX — 114 ops

<details>
<summary>Simple ops (direct WebNN mapping)</summary>

Abs, Add, And, ArgMax, ArgMin, AveragePool, BatchNormalization, Cast, Ceil, Clip, Concat, Conv, ConvInteger, ConvTranspose, Cos, CumSum, DepthToSpace, DequantizeLinear, Div, Dropout, DynamicQuantizeLinear, Elu, Equal, Erf, Exp, Expand, Flatten, Floor, Gather, GatherElements, GatherND, Gelu, Gemm, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, Greater, GreaterOrEqual, GroupNormalization, HardSigmoid, HardSwish, Identity, InstanceNormalization, IsInf, IsNaN, LRN, LayerNormalization, LeakyRelu, Less, LessOrEqual, Log, LpPool, MatMul, MatMulInteger, Max, MaxPool, Mean, Min, Mul, Neg, Not, Or, PRelu, Pad, Pow, QuantizeLinear, Reciprocal, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare, Relu, Reshape, Resize, Round, ScatterElements, ScatterND, Shape, Sigmoid, Sign, SimplifiedLayerNormalization, Sin, SkipSimplifiedLayerNormalization, Slice, Softmax, Softplus, Softsign, SpaceToDepth, Split, Sqrt, Squeeze, Sub, Tan, Tanh, Tile, Transpose, Trilu, Unsqueeze, Where, Xor

</details>

<details>
<summary>Composite ops (decomposed into WebNN primitives)</summary>

| Op | Strategy |
|----|----------|
| `GroupQueryAttention` | split → rotary embed → KV-cache concat → group broadcast → scaled dot-product attention |
| `MultiHeadAttention` | reshape → transpose → past-KV concat → scaled dot-product attention |
| `RotaryEmbedding` | split → cos/sin element-wise multiply → concat |
| `MatMulNBits` | uint4 dequantize → reshape → transpose → matmul |
| `GatherBlockQuantized` | block dequantize → gather |
| `LSTM` | `builder.lstm()` with bias splitting and initial-state handling |
| `GRU` | `builder.gru()` with zrn layout and reset-after semantics |
| `Einsum` | equation parsing → matmul / transpose / reshape |

</details>

### TFLite — 96 ops

<details>
<summary>Full list</summary>

ABS, ADD, ADD_N, ARG_MAX, ARG_MIN, AVERAGE_POOL_2D, BATCH_MATMUL, BROADCAST_TO, CAST, CEIL, CONCATENATION, CONV_2D, COS, CUMSUM, DEPTH_TO_SPACE, DEPTHWISE_CONV_2D, DEQUANTIZE, DIV, ELU, EQUAL, EXP, EXPAND_DIMS, FLOOR, FLOOR_DIV, FULLY_CONNECTED, GATHER, GATHER_ND, GELU, GREATER, GREATER_EQUAL, HARD_SWISH, L2_NORMALIZATION, L2_POOL_2D, LEAKY_RELU, LESS, LESS_EQUAL, LOCAL_RESPONSE_NORMALIZATION, LOG, LOGICAL_AND, LOGICAL_NOT, LOGICAL_OR, LOGISTIC, LOG_SOFTMAX, MAXIMUM, MAX_POOL_2D, MEAN, MINIMUM, MIRROR_PAD, MUL, NEG, NOT_EQUAL, PACK, PAD, PADV2, POW, PRELU, QUANTIZE, REDUCE_ALL, REDUCE_ANY, REDUCE_MAX, REDUCE_MIN, REDUCE_PROD, RELU, RELU6, RELU_0_TO_1, RELU_N1_TO_1, RESHAPE, RESIZE_BILINEAR, RESIZE_NEAREST_NEIGHBOR, ROUND, RSQRT, SCATTER_ND, SELECT, SELECT_V2, SHAPE, SIGN, SIN, SLICE, SOFTMAX, SPACE_TO_DEPTH, SPLIT, SPLIT_V, SQRT, SQUARE, SQUARED_DIFFERENCE, SQUEEZE, STRIDED_SLICE, SUB, SUM, TANH, TILE, TRANSPOSE, TRANSPOSE_CONV, UNPACK, WHERE, ZEROS_LIKE

</details>

### Unsupported ops

When a model contains ops with no WebNN equivalent (e.g. `TopK`, `Range`, `Mod`), the codegen:

1. Marks those op outputs as **dead**
2. Propagates dead state through all downstream ops automatically
3. Exports the last live **frontier tensors** before the dead zone instead of the original outputs

The generated graph always builds successfully. See [doc/unsupported-ops-bkm.md](doc/unsupported-ops-bkm.md) for details.

---

## Free dimension overrides

Models with symbolic dimensions (e.g. `batch_size`, `sequence_length`) are converted immediately with those names preserved. Override them to generate code with concrete shapes:

```bash
# CLI
npx tsx src/cli.ts model.onnx -d batch_size=1 -d sequence_length=128
```

In the web UI, input fields appear for each symbolic dimension. Entering a value auto-regenerates the code. Leaving a field empty keeps the symbolic name.

When using URL parameters, add `&dim=name:value` for each dimension (see [Web UI](#web-ui) above). Common examples:

| Model type | Typical overrides |
|------------|------------------|
| CLIP text encoder | `batch_size:1`, `sequence_length:77` |
| LLM / transformer | `batch_size:1`, `sequence_length:128` |
| Image classifier | `batch_size:1` |

See [ONNX Runtime freeDimensionOverrides](https://webnn.io/en/learn/tutorials/onnx-runtime/free-dimension-overrides).

---

## Architecture

```
.onnx / .tflite
      │
      ▼
   Parser          GraphIR        Op Registry        Codegen
  onnx.ts    ──▶  (shared)  ──▶  (per op/format) ──▶  javascript.ts
  tflite.ts                                            typescript.ts
                                                       html.ts
                                                          │
                              ┌───────────────┬──────────┴────────────┐
                              ▼               ▼                       ▼
                          model.js     model.weights          model.html
                                       model.manifest.json
```

### Directory structure

```
src/
├── index.ts          Public API — convert(), validateOperatorCoverage()
├── cli.ts            CLI entry point
├── ir/graph.ts       GraphIR, NodeIR, TensorInfo, ConstantInfo
├── parsers/
│   ├── onnx.ts       ONNX protobuf → GraphIR (external data supported)
│   └── tflite.ts     TFLite flatbuffers → GraphIR
├── operators/
│   ├── registry.ts   format + opType → emitter dispatch
│   ├── onnx/         Op builders, ported 1:1 from ORT WebNN provider
│   └── tflite/       TFLite op builders
├── weights/packer.ts WGWT binary + manifest generation
├── codegen/
│   ├── javascript.ts .js output — WeightsFile class + buildGraph()
│   ├── typescript.ts .ts output — adds WebNN type declarations
│   └── html.ts       .html output — runnable test harness
└── web/
    ├── app.ts        upload → convert → preview → download
    ├── upload.ts     file / folder / URL ingestion
    ├── preview.ts    Monaco editor preview
    └── download.ts   per-file and ZIP download
```

---

## Development

```bash
npm install      # install dependencies
npm run dev      # Vite dev server with HMR
npm run build    # production build → dist/
npm run lint     # tsc --noEmit
npm test         # vitest
```

### Adding an ONNX operator

1. Find the C++ builder in `reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/`
2. Port the logic to TypeScript in `src/operators/onnx/`
3. Register: `registerOnnxOp('OpName', emitterFn)` or `registerOnnxOps([…], emitterFn)`

The op is immediately available in the CLI, web UI, and library API.

### Adding an output format

Add a file to `src/codegen/` that accepts `GraphIR` and returns a `string`. Wire it into `convert()` in `src/index.ts`.

---

## Design notes

| Decision | Rationale |
|----------|-----------|
| No NCHW ↔ NHWC conversion | Chromium folds transpose constants at graph build time ([CL #6774969](https://chromium-review.googlesource.com/c/chromium/src/+/6774969)), matching ORT's removal of NHWC preferred layout ([PR #25679](https://github.com/microsoft/onnxruntime/pull/25679)) |
| Port ORT builders 1:1 | Reuses battle-tested attribute defaults and edge-case handling from production ORT |
| WGWT binary format | Minimal header + raw tensor data; no per-tensor framing overhead |
| Format-agnostic GraphIR | One shared IR for all parsers and codegen backends — adding a format means writing one parser |
| Client-side only | No server; models never leave the browser |
| Vanilla TypeScript + Vite | No UI framework; deploys as a single static site |

---

## Browser requirements

WebNN requires **Chrome 147+** with:

```
chrome://flags/#web-machine-learning-neural-network → Enabled
```

Devices: CPU, GPU, NPU via `MLDeviceType`.

> **Note:** `MLDeviceType` (`'cpu' | 'gpu' | 'npu'`) is a Chromium extension not in the [published WebNN spec](https://www.w3.org/TR/webnn/). Generated code includes it for compatibility with current Chrome builds.

---

## References

- [WebNN API specification](https://www.w3.org/TR/webnn/)
- [ORT WebNN Execution Provider](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/webnn)
- [ONNX operator schemas](https://onnx.ai/onnx/operators/)
- [TFLite built-in ops](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/converter/schema/schema.fbs)
- [WebNN Graph](https://github.com/rustnn/webnn-graph)
