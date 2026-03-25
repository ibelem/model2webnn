# model2webnn

Convert `.onnx` and `.tflite` models into ready-to-run [WebNN API](https://www.w3.org/TR/webnn/) JavaScript code.

[**Try it online**](https://ibelem.github.io/model2webnn) | [WebNN Spec](https://www.w3.org/TR/webnn/) | [WebNN Netron](https://ibelem.github.io/netron/)

## Overview

model2webnn parses ML model files and generates self-contained JavaScript that uses the WebNN `MLGraphBuilder` API. All processing runs client-side — models never leave the browser.

**Supported inputs:** ONNX (`.onnx`), TFLite (`.tflite`)
**Outputs:** `.js` code, `.weights` binary, `.manifest.json`, optional `.html` test page

## Getting started

### Web UI

1. Open [ibelem.github.io/model2webnn](https://ibelem.github.io/model2webnn)
2. Upload a model file or paste a HuggingFace URL
3. Code generates instantly — set free dimension overrides to regenerate with fixed values
4. Download the bundle from the header

**Direct link:** Append `?url=` to auto-fetch a model:
```
https://ibelem.github.io/model2webnn?url=https://huggingface.co/webnn/mobilenet-v2/resolve/main/onnx/model_fp16.onnx
```

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
| `--list-ops` | Print all supported operations |

### Library API

```typescript
import { convert } from 'model2webnn';

const buffer = new Uint8Array(await file.arrayBuffer());
const result = await convert(buffer, {
  format: 'javascript',
  freeDimensionOverrides: { batch_size: 1 },
});

result.code;      // JS source with buildGraph()
result.weights;   // Uint8Array — WGWT binary
result.manifest;  // { format, version, tensors: { ... } }
result.html;      // Self-contained test page
result.coverage;  // { totalOps, supportedOps, unsupportedOpTypes, ... }
```

## Output files

| File | Content |
|------|---------|
| `model.js` | `buildGraph(context, weights)` using `MLGraphBuilder` calls |
| `model.weights` | WGWT v1 binary — `"WGWT"` magic + version u32 + raw tensor data |
| `model.manifest.json` | Tensor metadata: data types, shapes, byte offsets |
| `model.html` | Runnable page with device selector and inference harness |

## Operator coverage

All 107 ONNX ops from the [ORT WebNN Execution Provider](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/webnn) are implemented, plus 7 additional ops. 95 TFLite ops are also supported.

### ONNX (114 ops)

<details>
<summary>Direct mappings (99 ops)</summary>

Abs, Add, And, ArgMax, ArgMin, AveragePool, BatchNormalization, Cast, Ceil, Clip, Concat, Conv, ConvInteger, ConvTranspose, Cos, CumSum, DepthToSpace, DequantizeLinear, Div, Dropout, DynamicQuantizeLinear, Elu, Equal, Erf, Exp, Expand, Flatten, Floor, Gather, GatherBlockQuantized, GatherElements, GatherND, Gelu, Gemm, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, Greater, GreaterOrEqual, GroupNormalization, HardSigmoid, HardSwish, Identity, InstanceNormalization, IsInf, IsNaN, LRN, LayerNormalization, LeakyRelu, Less, LessOrEqual, Log, LpPool, MatMul, MatMulInteger, Max, MaxPool, Mean, Min, Mul, Neg, Not, Or, PRelu, Pad, Pow, QuantizeLinear, Reciprocal, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare, Relu, Reshape, Resize, Round, ScatterElements, ScatterND, Shape, Sigmoid, Sign, SimplifiedLayerNormalization, Sin, SkipSimplifiedLayerNormalization, Slice, Softmax, Softplus, Softsign, SpaceToDepth, Split, Sqrt, Squeeze, Sub, Tan, Tanh, Tile, Transpose, Trilu, Unsqueeze, Where, Xor

</details>

<details>
<summary>Composite ops (8 ops, decomposed into WebNN primitives)</summary>

| Op | Decomposition |
|----|--------------|
| GroupQueryAttention | split → rotary → KV cache → group broadcast → SDPA |
| MultiHeadAttention | reshape → transpose → past concat → SDPA |
| RotaryEmbedding | split → cos/sin multiply → add → concat |
| MatMulNBits | dequantize(uint4) → reshape → transpose → matmul |
| LSTM | `builder.lstm()` with bias splitting and state management |
| GRU | `builder.gru()` with zrn layout and reset-after |
| Einsum | equation parsing → matmul/transpose/reshape |
| GatherBlockQuantized | dequantize → gather |

</details>

<details>
<summary>Quantization-aware ops</summary>

| Op | WebNN call |
|----|-----------|
| DequantizeLinear | `builder.dequantizeLinear(input, scale, zeroPoint)` |
| QuantizeLinear | `builder.quantizeLinear(input, scale, zeroPoint)` |
| DynamicQuantizeLinear | reduceMin/Max → scale/zp computation → quantizeLinear |
| MatMulNBits | 4-bit dequant → matmul |
| GatherBlockQuantized | block dequant → gather |
| ConvInteger | `builder.conv2d()` (quantized inputs) |
| MatMulInteger | `builder.matmul()` (quantized inputs) |

</details>

### TFLite (95 ops)

<details>
<summary>Full list</summary>

ABS, ADD, ADD_N, ARG_MAX, ARG_MIN, AVERAGE_POOL_2D, BATCH_MATMUL, BROADCAST_TO, CAST, CEIL, CONCATENATION, CONV_2D, COS, CUMSUM, DEPTH_TO_SPACE, DEPTHWISE_CONV_2D, DEQUANTIZE, DIV, ELU, EQUAL, EXP, EXPAND_DIMS, FLOOR, FLOOR_DIV, FULLY_CONNECTED, GATHER, GATHER_ND, GELU, GREATER, GREATER_EQUAL, HARD_SWISH, L2_NORMALIZATION, LEAKY_RELU, LESS, LESS_EQUAL, LOG, LOGICAL_AND, LOGICAL_NOT, LOGICAL_OR, LOG_SOFTMAX, LOGISTIC, MAX_POOL_2D, MAXIMUM, MEAN, MINIMUM, MIRROR_PAD, MUL, NEG, NOT_EQUAL, PACK, PAD, PADV2, POW, PRELU, QUANTIZE, RANGE, REDUCE_ANY, REDUCE_MAX, REDUCE_MIN, REDUCE_PROD, RELU, RELU6, RESHAPE, RESIZE_BILINEAR, RESIZE_NEAREST_NEIGHBOR, RSQRT, ROUND, SELECT_V2, SIN, SLICE, SOFTMAX, SPACE_TO_DEPTH, SPLIT, SPLIT_V, SQRT, SQUARE, SQUEEZE, STRIDED_SLICE, SUB, SUM, TANH, TILE, TRANSPOSE, UNPACK, WHERE

</details>

## Free dimension overrides

Models with symbolic dimensions (e.g. `batch_size`, `sequence_length`) can keep their symbolic names in the generated code, or be overridden to fixed integer values.

```bash
# CLI
npx tsx src/cli.ts model.onnx -d batch_size=1 -d sequence_length=128
```

In the web UI, code generates immediately with symbolic dims. Input fields appear for each symbolic dimension — typing a value auto-regenerates the code with that override applied. Empty fields keep the symbolic name as-is.

See: [ONNX Runtime freeDimensionOverrides](https://webnn.io/en/learn/tutorials/onnx-runtime/free-dimension-overrides)

## Weight format (WGWT)

```
Offset  Content
0–3     "WGWT"  (magic bytes, ASCII)
4–7     1       (version, little-endian u32)
8+      raw tensor data (concatenated)
```

The companion `.manifest.json` maps tensor names to their location:

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

## Architecture

```
.onnx / .tflite
      │
      ▼
┌──────────┐    ┌──────────┐    ┌────────────┐    ┌──────────────┐
│  Parser   │───▶│ GraphIR  │───▶│  Operator  │───▶│   Codegen    │
│ onnx.ts   │    │ (shared) │    │  Registry  │    │ javascript.ts│
│ tflite.ts │    └──────────┘    └────────────┘    │ typescript.ts│
└──────────┘                                       │ html.ts      │
                                                   └──────┬───────┘
                                                          │
                                          ┌───────────────┼───────────────┐
                                          ▼               ▼               ▼
                                      model.js      model.weights   model.html
                                                   model.manifest.json
```

### Source layout

```
src/
├── index.ts            Public API: convert(), validateOperatorCoverage()
├── cli.ts              CLI entry point
├── ir/
│   └── graph.ts        GraphIR, NodeIR, TensorInfo, ConstantInfo types
├── parsers/
│   ├── onnx.ts         ONNX protobuf → GraphIR (with external data support)
│   └── tflite.ts       TFLite flatbuffers → GraphIR
├── operators/
│   ├── registry.ts     Op dispatch: format + opType → emitter function
│   ├── onnx/           18 builder files, ported 1:1 from ORT C++
│   └── tflite/         TFLite op builders
├── weights/
│   └── packer.ts       WGWT binary packing + manifest generation
├── codegen/
│   ├── javascript.ts   Emit .js with WeightsFile class + buildGraph()
│   ├── typescript.ts   Emit .ts with WebNN type declarations
│   └── html.ts         Emit self-contained runnable .html
└── web/
    ├── index.html      Web UI shell
    ├── app.ts           Orchestration: upload → parse → preview → download
    ├── upload.ts        File/folder/URL upload + external data handling
    ├── preview.ts       Monaco editor code preview
    └── download.ts      ZIP bundle download
```

## Development

```bash
npm install          # Install dependencies
npm run dev          # Start dev server (Vite)
npm run build        # Production build
npm run lint         # Type-check (tsc --noEmit)
npm test             # Run tests (vitest)
```

### Adding a new ONNX operator

1. Find the ORT builder in `reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/`
2. Port the logic to a TypeScript emitter function in `src/operators/onnx/`
3. Register with `registerOnnxOp('OpName', emitterFn)` or `registerOnnxOps([...], emitterFn)`
4. The op is automatically available in CLI, Web UI, and library API

### Adding a new output format

Create a new file in `src/codegen/` that accepts `GraphIR` and returns a string. Wire it into `convert()` in `src/index.ts`.

## Deployment

**GitHub Pages** — Push to `main`. GitHub Actions runs `npm run build` and deploys `dist/` to `gh-pages`.

**Vercel** — Connect the repo. Vite is auto-detected. Build output: `dist/`.

## Design decisions

| Decision | Rationale |
|----------|-----------|
| No NCHW↔NHWC conversion | Chromium handles transpose constant folding (CL [#6774969](https://chromium-review.googlesource.com/c/chromium/src/+/6774969)). ORT [PR #25679](https://github.com/microsoft/onnxruntime/pull/25679) removed NHWC preferred layout. |
| Port ORT builders 1:1 | 107 ops battle-tested in production ONNX Runtime. Same attribute defaults and edge-case handling. |
| WGWT weight format | Purpose-built for WebNN with validation magic header |
| Format-agnostic IR | Single `GraphIR` type shared by all parsers and code generators. Adding a format = writing one parser. |
| Client-side only | Models never leave the browser. No server required. |
| Vanilla TS + Vite | Zero framework dependencies. Fast builds. Deploys as a static site. |

## Browser compatibility

WebNN requires Chrome 131+ with the following flags enabled:

- `chrome://flags/#web-machine-learning-neural-network` → Enabled
- Device support: CPU, GPU, NPU (via `MLDeviceType`)

> **Note:** `MLDeviceType` (`'cpu'` | `'gpu'` | `'npu'`) is implemented in Chromium but removed from the [published spec](https://www.w3.org/TR/webnn/). Generated code includes it for current Chrome builds.

## References

- [WebNN API Specification](https://www.w3.org/TR/webnn/)
- [WebNN Execution Provider of ONNX Runtime](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/webnn)