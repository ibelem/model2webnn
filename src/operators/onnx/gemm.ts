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
    // DEVIATION from ORT: use cast+sub instead of dequantizeLinear, because A/B may be outputs of
    // DynamicQuantizeLinear on tensors whose rank is absent from value_info (unranked intermediate
    // tensors). The correct scale/zp shape is unknowable at codegen time, so we can't satisfy
    // Chromium's requirement that scale rank == input rank.
    // cast+sub is equivalent when scale=1.0: dequantize(x, 1.0, zp) ≡ cast(x) − cast(zp).
    const output = emitter.declare(node.outputs[0]);

    // Dequantize A: cast(a, float32) - cast(a_zero_point, float32)
    const hasAZp = node.inputs.length > 2 && node.inputs[2] !== '';
    const castA = `${output}_cast_a`;
    emitter.line(`const ${castA} = builder.cast(${a}, 'float32');`);
    const dequantA = `${output}_dequant_a`;
    if (hasAZp) {
      const aZpRaw = emitter.ref(node.inputs[2]);
      const castAZp = `${output}_cast_a_zp`;
      emitter.line(`const ${castAZp} = builder.cast(${aZpRaw}, 'float32');`);
      emitter.line(`const ${dequantA} = builder.sub(${castA}, ${castAZp});`);
    } else {
      emitter.line(`const ${dequantA} = ${castA};`);
    }

    // Dequantize B: cast(b, float32) - cast(b_zero_point, float32)
    const hasBZp = node.inputs.length > 3 && node.inputs[3] !== '';
    const castB = `${output}_cast_b`;
    emitter.line(`const ${castB} = builder.cast(${b}, 'float32');`);
    const dequantB = `${output}_dequant_b`;
    if (hasBZp) {
      const bZpRaw = emitter.ref(node.inputs[3]);
      const castBZp = `${output}_cast_b_zp`;
      emitter.line(`const ${castBZp} = builder.cast(${bZpRaw}, 'float32');`);
      emitter.line(`const ${dequantB} = builder.sub(${castB}, ${castBZp});`);
    } else {
      emitter.line(`const ${dequantB} = ${castB};`);
    }

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
