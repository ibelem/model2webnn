// Ported from: reference/microsoft/onnxruntime/core/providers/webnn/builders/impl/einsum_op_builder.cc
// ONNX Einsum → WebNN reshape + transpose + matmul (for 2-input case)

import type { NodeIR } from '../../ir/graph.js';
import type { CodeEmitter } from '../registry.js';
import { registerOnnxOp } from '../registry.js';

function emitEinsum(node: NodeIR, emitter: CodeEmitter): void {
  const equation = (node.attributes.equation as string) ?? '';
  const output = emitter.declare(node.outputs[0]);

  emitter.comment(`Einsum: ${equation}`);

  if (node.inputs.length === 1) {
    // Single input — trace/sum along diagonal
    const input = emitter.ref(node.inputs[0]);
    emitter.comment('Single-input einsum — identity or trace');
    emitter.line(`const ${output} = ${input}; // Einsum: ${equation}`);
    return;
  }

  if (node.inputs.length !== 2) {
    emitter.comment(`Einsum with ${node.inputs.length} inputs — decomposition not supported`);
    emitter.line(`const ${output} = ${emitter.ref(node.inputs[0])}; // UNSUPPORTED: ${node.inputs.length}-input Einsum`);
    return;
  }

  const a = emitter.ref(node.inputs[0]);
  const b = emitter.ref(node.inputs[1]);

  // Parse equation: "ij,jk->ik"
  const parts = equation.replace(/\s/g, '').split('->');
  if (parts.length !== 2) {
    emitter.line(`const ${output} = builder.matmul(${a}, ${b}); // Einsum: ${equation} (fallback)`);
    return;
  }

  const [inputsPart, outputPart] = parts;
  const [labelsA, labelsB] = inputsPart.split(',');
  if (!labelsA || !labelsB) {
    emitter.line(`const ${output} = builder.matmul(${a}, ${b}); // Einsum: ${equation} (fallback)`);
    return;
  }

  // Classify dimensions per ORT logic:
  // (in_a, in_b, in_out) → batch(1,1,1), a_free(1,0,1), b_free(0,1,1), contract(1,1,0), sum_a(1,0,0), sum_b(0,1,0)
  const allLabels = new Set([...labelsA, ...labelsB, ...outputPart]);
  const batchLabels: string[] = [];
  const contractLabels: string[] = [];
  const aFreeLabels: string[] = [];
  const bFreeLabels: string[] = [];

  for (const l of allLabels) {
    const inA = labelsA.includes(l);
    const inB = labelsB.includes(l);
    const inOut = outputPart.includes(l);
    if (inA && inB && inOut) batchLabels.push(l);
    else if (inA && inB && !inOut) contractLabels.push(l);
    else if (inA && !inB && inOut) aFreeLabels.push(l);
    else if (!inA && inB && inOut) bFreeLabels.push(l);
    // sum dims (1,0,0) and (0,1,0) are reduced
  }

  // Common patterns — use matmul directly when possible
  const isSimpleMatmul = batchLabels.length === 0 && contractLabels.length === 1
    && aFreeLabels.length === 1 && bFreeLabels.length === 1;

  const isBatchedMatmul = contractLabels.length === 1
    && aFreeLabels.length === 1 && bFreeLabels.length === 1;

  if (isSimpleMatmul || isBatchedMatmul) {
    // Check if we need transpose
    // Standard matmul: ...ij,jk->...ik (contract dim is last of A, second-to-last of B)
    const contractIdx_A = labelsA.indexOf(contractLabels[0]);
    const contractIdx_B = labelsB.indexOf(contractLabels[0]);

    let aVar = a;
    let bVar = b;

    // If contract dim is not last dim of A, transpose
    if (contractIdx_A !== labelsA.length - 1) {
      const perm = [...Array(labelsA.length).keys()];
      perm.splice(contractIdx_A, 1);
      perm.push(contractIdx_A);
      const aT = `${output}_a_t`;
      emitter.line(`const ${aT} = builder.transpose(${a}, { permutation: [${perm}] });`);
      aVar = aT;
    }

    // If contract dim is not second-to-last of B (for >=2D) or first dim (for 2D)
    const expectedBContractIdx = labelsB.length >= 2 ? labelsB.length - 2 : 0;
    if (contractIdx_B !== expectedBContractIdx) {
      const perm = [...Array(labelsB.length).keys()];
      perm.splice(contractIdx_B, 1);
      perm.splice(expectedBContractIdx, 0, contractIdx_B);
      const bT = `${output}_b_t`;
      emitter.line(`const ${bT} = builder.transpose(${b}, { permutation: [${perm}] });`);
      bVar = bT;
    }

    emitter.line(`const ${output} = builder.matmul(${aVar}, ${bVar});`);
    return;
  }

  // Fallback: general case — just emit matmul with a comment
  emitter.comment(`Complex einsum decomposition — falling back to matmul`);
  emitter.line(`const ${output} = builder.matmul(${a}, ${b}); // Einsum: ${equation}`);
}

registerOnnxOp('Einsum', emitEinsum);
