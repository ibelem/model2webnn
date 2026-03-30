# BKM: Explicit Padding for WebNN Conv2D and Pool2D

## Summary

WebNN does **not** support `autoPad`. All convolution and pooling operations must use explicit `padding: [top, bottom, left, right]` values computed at code-generation time.

## Problem

TFLite models declare `SAME` or `VALID` padding on convolutions and pooling ops. An earlier version of the code generator emitted `autoPad: 'same-upper'` in the generated WebNN JavaScript, assuming it was a valid option. This caused **all** TFLite models with SAME-padded ops to fail at graph build time:

```
Failed to execute 'add' on 'MLGraphBuilder': The input shapes are not broadcastable.
```

### Why the error says "add" and not "conv2d"

Chromium's WebIDL bindings silently ignore unknown dictionary members. When `autoPad: 'same-upper'` is passed to `builder.conv2d(...)`, it is simply discarded — the conv2d still executes, but with zero padding. This produces an output tensor with the **wrong spatial dimensions** (smaller than expected). When a downstream `builder.add()` tries to element-wise add this wrongly-sized tensor with a correctly-sized residual branch, the shapes don't match and the broadcast check fails.

This is why the error message is misleading — the real bug is in the conv2d/pool2d padding, not in the add op itself.

### Root cause confirmation

- The WebNN spec ([`MLConv2dOptions`](https://www.w3.org/TR/webnn/#dictdef-mlconv2doptions)) only defines `padding`, `strides`, `dilations`, `groups`, `inputLayout`, `filterLayout`, and `bias`. There is no `autoPad` property.
- The Chromium WebNN IDL (`webnn_graph_builder.mojom`) has no `autoPad` field.
- The Chromium source (`webnn_graph_builder_impl.cc`) computes output shapes using only the explicit `padding` array.

## Solution

Compute explicit padding values at code-generation time using the input tensor shape, kernel size, strides, and dilations. The formula follows TFLite's SAME padding convention (same as ONNX's `SAME_UPPER`):

```
outputSize = ceil(inputSize / stride)
totalPad   = max(0, (outputSize - 1) * stride + effectiveKernel - inputSize)
padBefore  = floor(totalPad / 2)
padAfter   = totalPad - padBefore
```

where `effectiveKernel = (kernel - 1) * dilation + 1`.

### Implementation

A shared helper in `src/operators/tflite/index.ts`:

```typescript
function computeSamePadding(
  inputH: number, inputW: number,
  kernelH: number, kernelW: number,
  strideH: number, strideW: number,
  dilationH: number, dilationW: number
): [number, number, number, number] {
  const effectiveKH = (kernelH - 1) * dilationH + 1;
  const effectiveKW = (kernelW - 1) * dilationW + 1;
  const outputH = Math.ceil(inputH / strideH);
  const outputW = Math.ceil(inputW / strideW);
  const totalPadH = Math.max(0, (outputH - 1) * strideH + effectiveKH - inputH);
  const totalPadW = Math.max(0, (outputW - 1) * strideW + effectiveKW - inputW);
  const padTop = Math.floor(totalPadH / 2);
  const padBottom = totalPadH - padTop;
  const padLeft = Math.floor(totalPadW / 2);
  const padRight = totalPadW - padLeft;
  return [padTop, padBottom, padLeft, padRight];
}
```

### Emitter usage

```typescript
if (padding === 'SAME') {
  const inputShape = emitter.tensorShape(node.inputs[0]);  // NHWC
  const filterShape = emitter.constantShape(node.inputs[1]); // OHWI or IHWO
  if (inputShape && typeof inputShape[1] === 'number' && typeof inputShape[2] === 'number') {
    const [t, b, l, r] = computeSamePadding(
      inputShape[1] as number, inputShape[2] as number,
      filterShape[kHIndex], filterShape[kWIndex],
      strides[0], strides[1], dilations[0], dilations[1]);
    opts.push(`padding: [${t}, ${b}, ${l}, ${r}]`);
  }
}
```

## Affected Operations

| Emitter | Filter layout | Kernel indices (H, W) |
|---------|--------------|----------------------|
| `emitConv2D` | `ohwi` | `[1], [2]` |
| `emitDepthwiseConv2D` | `ihwo` | `[1], [2]` |
| `emitTransposeConv` | `ohwi` | `[1], [2]` |
| `emitPool2D` | — | `windowDimensions[0], [1]` |
| `emitL2Pool2D` | — | `windowDimensions[0], [1]` |

The same fix was applied to the ONNX emitters (`src/operators/onnx/conv.ts`, `src/operators/onnx/pool.ts`) for models that use `auto_pad: SAME_UPPER` or `SAME_LOWER`. ONNX uses NCHW layout by default, so the input spatial dims are at indices `[2], [3]` and the kernel shape is `oihw` (indices `[2], [3]`).

## Padding Order Convention

WebNN `padding` is `[beginning_height, ending_height, beginning_width, ending_width]` — i.e., `[top, bottom, left, right]`.

ONNX `pads` is `[top, left, bottom, right]` — note the different order.

When converting ONNX explicit pads to WebNN: `padding: [pads[0], pads[2], pads[1], pads[3]]`.

## Verification Examples

| Operation | Input (NHWC) | Kernel | Stride | Expected Output | Computed Padding |
|-----------|-------------|--------|--------|----------------|-----------------|
| Conv2D SAME | `[1,300,300,3]` | `3×3` | `2×2` | `[1,150,150,32]` | `[0, 1, 0, 1]` |
| DWConv2D SAME | `[1,150,150,32]` | `3×3` | `1×1` | `[1,150,150,32]` | `[1, 1, 1, 1]` |
| Conv2D SAME | `[1,150,150,32]` | `1×1` | `1×1` | `[1,150,150,24]` | `[0, 0, 0, 0]` |
| DWConv2D SAME | `[1,75,75,192]` | `5×5` | `2×2` | `[1,38,38,192]` | `[1, 2, 1, 2]` |

## Key Takeaway

Never emit `autoPad` in generated WebNN code. Always resolve padding to explicit numeric values during code generation. Chromium will silently discard any unknown property in option dictionaries, making bugs extremely hard to trace.
