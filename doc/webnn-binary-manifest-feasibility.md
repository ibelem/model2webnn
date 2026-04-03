# Feasibility Analysis: `.webnn` Binary Manifest vs `.manifest.json`

## Current State

The manifest is a simple JSON like:

```json
{
  "format": "wg-weights-manifest",
  "version": 1,
  "endianness": "little",
  "tensors": {
    "weight_0": { "dataType": "float32", "shape": [32,3,3,3], "byteOffset": 8, "byteLength": 3456 }
  }
}
```

## Size Reduction: Minimal

The manifest is **metadata only** — tensor names, shapes, offsets, data types. For a typical model with ~100 tensors, this JSON is **2–10 KB**. Protobuf/FlatBuffers encoding might shrink it to 1–5 KB. The `.weights` binary file (the actual tensor data) is typically **10–500 MB**. The manifest is <0.01% of total download size — **binary encoding saves essentially nothing**.

## Reverse-Engineering Protection: Weak

A binary manifest only adds a trivial speed bump. The `.weights` file is raw tensor data — anyone can reconstruct shapes/types by inspecting the generated `.js` code, which contains all `builder.constant()` calls with explicit shapes and offsets. True obfuscation would require encrypting the weights themselves, which is a different problem entirely.

## Cost of the `.webnn` Approach

| Factor | Impact |
|---|---|
| **Extra JS library** | `protobufjs` is ~150 KB min+gzip, or flatbuffers ~15 KB. Either **dwarfs** the JSON savings. |
| **Parse latency** | `JSON.parse()` is native C++ in browsers — extremely fast. Protobuf/FlatBuffers JS decoders are slower for small payloads. |
| **Developer experience** | Manifest is no longer human-readable — harder to debug. |
| **Content-Type** | `application/octet-stream` (no standard MIME for `.webnn`). Browsers won't preview it. |
| **Complexity** | Two code paths (JSON vs binary), schema maintenance, extra codegen logic — significant ongoing cost. |

## Recommendation: Don't Do It

The manifest JSON is tiny relative to the weights. A binary format adds library overhead that exceeds the savings, makes debugging harder, and provides negligible reverse-engineering protection. The engineering cost (two code paths in upload, preview, codegen, download) is not justified.

If you want to reduce total download size, **compress the `.weights` file** (gzip/brotli via server config) — that will save orders of magnitude more than shrinking the manifest. If you want obfuscation, that's a separate feature requiring weight encryption, not just a binary manifest format.
