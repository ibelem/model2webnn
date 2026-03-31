// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/gemm_op_builder.cc
// Maps ONNX Gemm → matmul + optional operations, MatMul/MatMulInteger → matmul

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOps } from '../registry.js';

function emitGemm(node: NodeIR, emitter: CodeEmitter): void {
  const a = emitter.ref(node.inputs[0]);
  const b = emitter.ref(node.inputs[1]);

  if (node.opType === 'MatMul') {
    const output = emitter.declare(node.outputs[0]);
    emitter.line(`const ${output} = builder.matmul(${a}, ${b});`);
    return;
  }

  if (node.opType === 'MatMulInteger') {
    // ORT gemm_op_builder.cc: MatMulInteger decomposes to dequantizeLinear → matmul → cast(int32)
    // WebNN matmul only supports float32/float16 inputs.
    const output = emitter.declare(node.outputs[0]);

    // Dequantize A: dequantizeLinear(a, scale=1.0, a_zero_point)
    const aShape = emitter.tensorShape(node.inputs[0]);
    const aRank = aShape ? aShape.length : 2;
    const aTargetShape = Array(aRank).fill(1);

    let aZpRef: string;
    const hasAZp = node.inputs.length > 2 && node.inputs[2] !== '';
    if (hasAZp) {
      const aZpRaw = emitter.ref(node.inputs[2]);
      const aZpGraphShape = emitter.tensorShape(node.inputs[2]);
      const aZpRank = aZpGraphShape ? aZpGraphShape.length : 0;
      if (aZpRank === 1 && aShape && typeof aShape[0] === 'number' && aShape[0] !== 1) {
        // Per-row zero point for A: shape [M] → [M, 1, ...]
        aTargetShape[0] = aShape[0];
      }
      if (aZpRank !== aRank) {
        aZpRef = `${output}_a_zp_reshaped`;
        emitter.line(`const ${aZpRef} = builder.reshape(${aZpRaw}, [${aTargetShape.join(', ')}]);`);
      } else {
        aZpRef = aZpRaw;
      }
    } else {
      const aDtype = emitter.tensorDataType(node.inputs[0]) ?? 'uint8';
      const arrayType = aDtype === 'int8' ? 'Int8Array' : 'Uint8Array';
      aZpRef = `${output}_a_zp`;
      emitter.line(`const ${aZpRef} = builder.constant({dataType: '${aDtype}', shape: [${aTargetShape.join(', ')}]}, new ${arrayType}([0]));`);
    }
    const aScaleRef = `${output}_a_scale`;
    emitter.line(`const ${aScaleRef} = builder.constant({dataType: 'float32', shape: [${aTargetShape.join(', ')}]}, new Float32Array([1]));`);
    const dequantA = `${output}_dequant_a`;
    emitter.line(`const ${dequantA} = builder.dequantizeLinear(${a}, ${aScaleRef}, ${aZpRef});`);

    // Dequantize B: dequantizeLinear(b, scale=1.0, b_zero_point)
    const bShape = emitter.tensorShape(node.inputs[1]);
    const bRank = bShape ? bShape.length : 2;
    const bTargetShape = Array(bRank).fill(1);

    let bZpRef: string;
    const hasBZp = node.inputs.length > 3 && node.inputs[3] !== '';
    if (hasBZp) {
      const bZpRaw = emitter.ref(node.inputs[3]);
      const bZpGraphShape = emitter.tensorShape(node.inputs[3]);
      const bZpRank = bZpGraphShape ? bZpGraphShape.length : 0;
      if (bZpRank === 1 && bShape && bShape.length > 0) {
        // Per-column zero point for B: shape [K] → [..., K] (last dim)
        const lastDimVal = bShape[bShape.length - 1];
        if (typeof lastDimVal === 'number') {
          bTargetShape[bTargetShape.length - 1] = lastDimVal;
        }
      }
      if (bZpRank !== bRank) {
        bZpRef = `${output}_b_zp_reshaped`;
        emitter.line(`const ${bZpRef} = builder.reshape(${bZpRaw}, [${bTargetShape.join(', ')}]);`);
      } else {
        bZpRef = bZpRaw;
      }
    } else {
      const aDtype = emitter.tensorDataType(node.inputs[0]) ?? 'uint8';
      const arrayType = aDtype === 'int8' ? 'Int8Array' : 'Uint8Array';
      bZpRef = `${output}_b_zp`;
      emitter.line(`const ${bZpRef} = builder.constant({dataType: '${aDtype}', shape: [${bTargetShape.join(', ')}]}, new ${arrayType}([0]));`);
    }
    const bScaleRef = `${output}_b_scale`;
    emitter.line(`const ${bScaleRef} = builder.constant({dataType: 'float32', shape: [${bTargetShape.join(', ')}]}, new Float32Array([1]));`);
    const dequantB = `${output}_dequant_b`;
    emitter.line(`const ${dequantB} = builder.dequantizeLinear(${b}, ${bScaleRef}, ${bZpRef});`);

    // Matmul with float32 inputs → cast to int32
    const matmulOut = `${output}_matmul`;
    emitter.line(`const ${matmulOut} = builder.matmul(${dequantA}, ${dequantB});`);
    emitter.line(`const ${output} = builder.cast(${matmulOut}, 'int32');`);
    return;
  }

  // Gemm: alpha * A' * B' + beta * C
  const transA = (node.attributes.transA as number) ?? 0;
  const transB = (node.attributes.transB as number) ?? 0;
  const alpha = (node.attributes.alpha as number) ?? 1.0;
  const beta = (node.attributes.beta as number) ?? 1.0;
  const hasC = node.inputs.length > 2 && node.inputs[2] !== '';

  let aRef = a;
  let bRef = b;

  // Handle transposes
  if (transA) {
    const aT = emitter.declare(`${node.outputs[0]}_a_transposed`);
    emitter.line(`const ${aT} = builder.transpose(${a});`);
    aRef = aT;
  }
  if (transB) {
    const bT = emitter.declare(`${node.outputs[0]}_b_transposed`);
    emitter.line(`const ${bT} = builder.transpose(${b});`);
    bRef = bT;
  }

  // MatMul
  let result: string;
  const matmulOut = emitter.declare(`${node.outputs[0]}_matmul`);
  emitter.line(`const ${matmulOut} = builder.matmul(${aRef}, ${bRef});`);
  result = matmulOut;

  // Apply alpha if != 1.0
  if (alpha !== 1.0) {
    const alphaOut = emitter.declare(`${node.outputs[0]}_alpha`);
    emitter.line(`const ${alphaOut} = builder.mul(${result}, builder.constant({ dataType: 'float32', shape: [] }, new Float32Array([${alpha}])));`);
    result = alphaOut;
  }

  // Add bias (C) with optional beta
  if (hasC) {
    const c = emitter.ref(node.inputs[2]);
    let cRef = c;
    if (beta !== 1.0) {
      const betaOut = emitter.declare(`${node.outputs[0]}_beta`);
      emitter.line(`const ${betaOut} = builder.mul(${c}, builder.constant({ dataType: 'float32', shape: [] }, new Float32Array([${beta}])));`);
      cRef = betaOut;
    }
    const output = emitter.declare(node.outputs[0]);
    emitter.line(`const ${output} = builder.add(${result}, ${cRef});`);
  } else {
    const output = emitter.declare(node.outputs[0]);
    emitter.line(`const ${output} = ${result};`);
  }
}

registerOnnxOps(['Gemm', 'MatMul', 'MatMulInteger'], emitGemm);
