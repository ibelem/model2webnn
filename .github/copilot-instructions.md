# model2webnn — Copilot Instructions

## Project Overview

model2webnn converts ML models (.onnx, .tflite) into ready-to-run WebNN API JavaScript/TypeScript code. It runs both as a CLI tool and as a web application deployed to GitHub Pages / Vercel.

## Architecture

```
src/
├── index.ts              # Public library API
├── cli.ts                # CLI entry point (Node.js only)
├── ir/                   # Intermediate representation (format-agnostic)
│   └── graph.ts          # GraphIR, NodeIR, TensorInfo, ConstantInfo types
├── parsers/              # Model format parsers → GraphIR
│   ├── onnx.ts           # ONNX protobuf parser (protobufjs)
│   └── tflite.ts         # TFLite flatbuffers parser
├── operators/            # Operator mapping: model ops → WebNN builder calls
│   ├── registry.ts       # Central op dispatch registry
│   ├── onnx/             # ONNX op builders (1 file per ORT builder)
│   └── tflite/           # TFLite op builders
├── weights/              # Weight extraction and packing
│   └── packer.ts         # WGWT binary packing + manifest generation
├── codegen/              # Code generation backends
│   ├── javascript.ts     # Emit .js with MLGraphBuilder calls
│   ├── typescript.ts     # Emit .ts with @webnn/types annotations
│   └── html.ts           # Emit runnable .html test harness
└── web/                  # Web UI (Vite app)
    ├── index.html
    ├── app.ts
    ├── upload.ts          # File upload + URL fetch (HuggingFace etc.)
    ├── preview.ts         # Monaco editor code preview
    └── download.ts        # Bundle download (.html + .js + .weights)
```

## Critical Rules

### 1. Operator Implementations MUST Follow ORT Builders

All operator implementations must strictly follow the C++ op builder implementations in:
```
reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/
```

- Port the logic 1:1 from C++ to TypeScript
- Preserve attribute handling, default values, and edge cases
- For composite ops (GQA, MHA, RotaryEmbedding, MatMulNBits, LSTM, GRU, Einsum), replicate the full decomposition
- Only deviate if there is a clear bug or design issue — document any deviation with a comment explaining why

### 2. Layout Handling

Weight tensors are passed through as-is from the original model — no data reordering.
WebNN ops that support layout options must declare the correct layout matching the model's convention:

**ONNX models:**
- ONNX uses NCHW by default, which matches WebNN defaults — no layout options needed
- Reference: ONNX Runtime PR #25679, Chromium CL #6774969

**TFLite models (always NHWC):**
- `conv2d`: `inputLayout: 'nhwc'`, `filterLayout: 'ohwi'` (Conv2D) or `'ihwo'` (DepthwiseConv2D)
- `convTranspose2d`: `inputLayout: 'nhwc'`, `filterLayout: 'ohwi'`
- `averagePool2d` / `maxPool2d` / `l2Pool2d`: `layout: 'nhwc'`
- `resample2d`: `axes: [1, 2]` (spatial dims in NHWC)
- Element-wise ops (add, mul, relu, etc.) are layout-agnostic

### 3. Weight File Format: WGWT

Use the webnn-graph WGWT binary format for weight files:

**Binary `.weights` file:**
```
Bytes 0-3:   "WGWT"          (magic, ASCII)
Bytes 4-7:   0x01000000       (version 1, little-endian u32)
Bytes 8+:    [concatenated raw tensor data]
```

**Manifest `.manifest.json`:**
```json
{
  "format": "wg-weights-manifest",
  "version": 1,
  "endianness": "little",
  "tensors": {
    "tensor_name": {
      "dataType": "float32",
      "shape": [32, 3, 3, 3],
      "byteOffset": 8,
      "byteLength": 3456
    }
  }
}
```

### 4. Intermediate Representation

All parsers (ONNX, TFLite, etc.) must produce the same `GraphIR` structure:

```typescript
interface GraphIR {
  name: string;
  format: string;                    // "onnx" | "tflite"
  inputs: TensorInfo[];
  outputs: TensorInfo[];
  constants: ConstantInfo[];
  nodes: NodeIR[];
}

interface TensorInfo {
  name: string;
  dataType: string;                  // WebNN data types: "float32", "float16", "int8", etc.
  shape: (number | string)[];        // string for dynamic dims
}

interface ConstantInfo extends TensorInfo {
  rawData: Uint8Array;
  byteOffset?: number;               // filled during weight packing
  byteLength: number;
}

interface NodeIR {
  opType: string;                    // original model op name
  inputs: string[];                  // tensor names
  outputs: string[];
  attributes: Record<string, any>;
}
```

### 5. Code Style

- TypeScript strict mode, ES2020 target
- No classes unless genuinely needed — prefer functions and interfaces
- No layout conversion code (see rule 2)
- Use `protobufjs` for ONNX parsing, `flatbuffers` for TFLite
- All processing runs client-side in the web UI (no server)

### 6. Generated Code Style

Generated WebNN code should:
- Use `async function buildGraph(context, weightsFile)` as the entry point
- Include the `WeightsFile` helper class for loading .weights + .manifest.json
- Use descriptive variable names derived from model tensor names
- Include comments showing the original model op type
- Be self-contained and runnable in any modern browser with WebNN support

### 7. Web UI

- Vanilla TypeScript + Vite (no React/Vue/Angular)
- Monaco editor for code preview
- Must be deployable as static site to GitHub Pages and Vercel
- Support both file upload and URL input (HuggingFace model URLs)
- All model processing happens client-side

## Reference Code Locations

| Reference | Path | Use For |
|-----------|------|---------|
| ORT op builders | `reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/` | Op implementation logic (source of truth) |
| ORT model builder | `reference/microsoft/onnxruntime/core/providers/webnn/builders/model_builder.*` | Graph building patterns |
| WebNN spec (online) | https://www.w3.org/TR/webnn/ | Authoritative spec (latest) |
| WebNN spec IDL | `reference/webnn-spec/webnn.idl.txt` | API surface, data types, op signatures |
| WebNN spec text | `reference/webnn-spec/webnn.bs.txt` | Detailed op semantics |
| webnn-graph examples | `reference/webnn-graph-main/examples/` | WGWT format, WeightsFile loader, buildGraph patterns |
| webnn-code-generator | `reference/webnn-code-generator-main/src/` | Code generation patterns, UI patterns |
| TFLite schema (online) | https://github.com/google-ai-edge/LiteRT/blob/main/tflite/converter/schema/schema.fbs | TFLite FlatBuffers schema (BuiltinOperator, BuiltinOptions, option tables) |

## WebNN Data Type Mapping

### MLOperandDataType → ArrayBufferView

| `MLOperandDataType` | `ArrayBufferView` |
|--------------------|-----------------|
| float32 | Float32Array |
| float16 | Float16Array |
| int64 | BigInt64Array |
| uint64 | BigUint64Array |
| int32 | Int32Array |
| uint32 | Uint32Array |
| int8 | Int8Array |
| uint8 | Uint8Array |

### ONNX → WebNN Type Mapping

| ONNX Type ID | ONNX Name | WebNN `MLOperandDataType` |
|-------------|-----------|--------------------------|
| 1 | FLOAT | float32 |
| 2 | UINT8 | uint8 |
| 3 | INT8 | int8 |
| 5 | INT16 | *(unsupported — cast to int32)* |
| 6 | INT32 | int32 |
| 7 | INT64 | int64 |
| 10 | FLOAT16 | float16 |
| 11 | DOUBLE | float32 (downcast) |
| 12 | UINT32 | uint32 |
| 13 | UINT64 | uint64 |

### TFLite → WebNN Type Mapping

| TFLite Type | WebNN `MLOperandDataType` |
|------------|--------------------------|
| FLOAT32 | float32 |
| FLOAT16 | float16 |
| INT32 | int32 |
| UINT8 | uint8 |
| INT8 | int8 |
| INT64 | int64 |

## MLDeviceType (Chromium-only)

`MLDeviceType` was removed from the online WebNN spec (https://www.w3.org/TR/webnn/), but Chromium still uses it in its implementation. Generated code MUST include `MLDeviceType` support:

```typescript
// Chromium-only — not in the published spec, but required for current Chrome builds
type MLDeviceType = 'cpu' | 'gpu' | 'npu';
type MLPowerPreference = 'default' | 'high-performance' | 'low-power';

interface MLContextOptions {
  deviceType?: MLDeviceType;        // default: 'cpu'
  powerPreference?: MLPowerPreference; // default: 'default'
}
```

Generated `buildGraph` code should accept optional `deviceType` parameter and pass it to `navigator.ml.createContext()`.
