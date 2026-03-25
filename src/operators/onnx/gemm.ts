// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/gemm_op_builder.cc
// Maps ONNX Gemm → matmul + optional operations, MatMul/MatMulInteger → matmul

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOps } from '../registry.js';

function emitGemm(node: NodeIR, emitter: CodeEmitter): void {
  const a = emitter.ref(node.inputs[0]);
  const b = emitter.ref(node.inputs[1]);

  if (node.opType === 'MatMul' || node.opType === 'MatMulInteger') {
    const output = emitter.declare(node.outputs[0]);
    emitter.line(`const ${output} = builder.matmul(${a}, ${b});`);
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
