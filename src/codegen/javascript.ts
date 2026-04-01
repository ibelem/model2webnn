// JavaScript code generator — emits .js with MLGraphBuilder calls
// Produces a self-contained buildGraph function + WeightsFile helper.

import type { GraphIR, ConstantInfo, NodeIR, MLOperandDataType, TensorInfo } from '../ir/graph.js';
import { toJsVarName, getTypedArrayName } from '../ir/graph.js';
import type { CodeEmitter } from '../operators/registry.js';
import { getEmitter } from '../operators/registry.js';

// Import all ONNX ops to register them
import '../operators/onnx/index.js';
// Import all TFLite ops to register them
import '../operators/tflite/index.js';

export interface GenerateJsOptions {
  weightsFileName?: string; // default: "model.weights"
  manifestFileName?: string; // default: "model.manifest.json"
  includeWeightsLoader?: boolean; // default: true
}

/**
 * Find "frontier" live tensors — the last computed values before the dead zone.
 * These are live tensors that are consumed by at least one dead/unsupported node.
 * Used as fallback outputs when all original graph outputs are dead.
 */
function findFrontierTensors(graph: GraphIR, emitter: CodeEmitter): string[] {
  const graphInputNames = new Set(graph.inputs.map((i) => i.name));
  const constantNames = new Set(graph.constants.map((c) => c.name));
  const frontier = new Set<string>();

  for (const node of graph.nodes) {
    // Only look at dead/skipped nodes
    const anyOutputDead = node.outputs.some((o) => o !== '' && emitter.isDead(o));
    if (!anyOutputDead) continue;

    for (const inp of node.inputs) {
      if (inp === '') continue;
      if (emitter.isDead(inp)) continue; // already dead — not a frontier
      if (constantNames.has(inp)) continue; // skip weight tensors
      if (graphInputNames.has(inp)) continue; // skip graph inputs
      frontier.add(inp);
    }
  }

  return [...frontier];
}

export function generateJavaScript(
  graph: GraphIR,
  options: GenerateJsOptions = {},
): string {
  const {
    weightsFileName = 'model.weights',
    manifestFileName = 'model.manifest.json',
    includeWeightsLoader = true,
  } = options;

  const lines: string[] = [];
  const indent = '  ';
  let indentLevel = 0;

  function emit(line: string): void {
    if (line === '') {
      lines.push('');
    } else {
      lines.push(indent.repeat(indentLevel) + line);
    }
  }

  // --- WeightsFile helper class ---
  if (includeWeightsLoader) {
    emit('/**');
    emit(' * Helper class for loading and managing WebNN graph weights');
    emit(' * Format: WGWT v1 (webnn-graph weights)');
    emit(' */');
    emit('class WeightsFile {');
    indentLevel++;
    emit('constructor(buffer, manifest) {');
    indentLevel++;
    emit('this.buffer = buffer;');
    emit('this.manifest = manifest;');
    indentLevel--;
    emit('}');
    emit('');
    emit('static async load(weightsPath, manifestPath) {');
    indentLevel++;
    emit('const [weightsResponse, manifestResponse] = await Promise.all([');
    indentLevel++;
    emit('fetch(weightsPath),');
    emit('fetch(manifestPath),');
    indentLevel--;
    emit(']);');
    emit('');
    emit('if (!weightsResponse.ok) throw new Error(`Failed to load weights: ${weightsResponse.statusText}`);');
    emit('if (!manifestResponse.ok) throw new Error(`Failed to load manifest: ${manifestResponse.statusText}`);');
    emit('');
    emit('const buffer = await weightsResponse.arrayBuffer();');
    emit('const manifest = await manifestResponse.json();');
    emit('');
    emit('if (manifest.format !== \'wg-weights-manifest\') throw new Error(`Invalid manifest format: ${manifest.format}`);');
    emit('if (manifest.version !== 1) throw new Error(`Unsupported manifest version: ${manifest.version}`);');
    emit('');
    emit('const view = new DataView(buffer);');
    emit('const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));');
    emit('if (magic !== \'WGWT\') throw new Error(`Invalid weights file magic: ${magic}`);');
    emit('const version = view.getUint32(4, true);');
    emit('if (version !== 1) throw new Error(`Unsupported weights version: ${version}`);');
    emit('');
    emit('return new WeightsFile(buffer, manifest);');
    indentLevel--;
    emit('}');
    emit('');
    emit('getSlice(name) {');
    indentLevel++;
    emit('const tensor = this.manifest.tensors[name];');
    emit('if (!tensor) throw new Error(`Tensor not found: ${name}`);');
    emit('return tensor;');
    indentLevel--;
    emit('}');
    emit('');
    emit('getData(name) {');
    indentLevel++;
    emit('const t = this.getSlice(name);');
    emit('return this.buffer.slice(t.byteOffset, t.byteOffset + t.byteLength);');
    indentLevel--;
    emit('}');
    indentLevel--;
    emit('}');
    emit('');
  }

  // --- buildGraph function ---
  emit('/**');
  emit(` * Build WebNN graph for model: ${graph.name}`);
  emit(` * Source format: ${graph.format}`);
  emit(' * @param {MLContext} context - WebNN context');
  emit(` * @param {WeightsFile} weights - Loaded weights file`);
  emit(' * @returns {Promise<MLGraph>}');
  emit(' */');
  emit('async function buildGraph(context, weights) {');
  indentLevel++;
  emit('const builder = new MLGraphBuilder(context);');
  emit('');

  // Track variable declarations to avoid collisions
  const declaredVars = new Set<string>();
  const constantNames = new Set<string>();
  const constantMap = new Map<string, ConstantInfo>();

  for (const c of graph.constants) {
    constantNames.add(c.name);
    constantMap.set(c.name, c);
  }

  // Build the CodeEmitter implementation
  function makeVarName(tensorName: string): string {
    let name = toJsVarName(tensorName);
    if (declaredVars.has(name)) {
      let i = 2;
      while (declaredVars.has(`${name}_${i}`)) i++;
      name = `${name}_${i}`;
    }
    return name;
  }

  // Map from tensor name → JS variable name
  const varMap = new Map<string, string>();
  const deadTensors = new Set<string>();

  function getOrDeclare(tensorName: string): string {
    if (varMap.has(tensorName)) return varMap.get(tensorName)!;
    const name = makeVarName(tensorName);
    declaredVars.add(name);
    varMap.set(tensorName, name);
    return name;
  }

  const emitterImpl: CodeEmitter = {
    ref(tensorName: string): string {
      return getOrDeclare(tensorName);
    },
    declare(tensorName: string): string {
      return getOrDeclare(tensorName);
    },
    line(code: string): void {
      emit(code);
    },
    comment(text: string): void {
      emit(`// ${text}`);
    },
    isConstant(tensorName: string): boolean {
      return constantNames.has(tensorName);
    },
    constantShape(tensorName: string): number[] {
      const c = constantMap.get(tensorName);
      if (!c) return [];
      return c.shape.map((d) => (typeof d === 'number' ? d : 0));
    },
    constantDataType(tensorName: string): string {
      const c = constantMap.get(tensorName);
      if (!c) return 'float32';
      return c.dataType;
    },
    constantRawData(tensorName: string): Uint8Array | null {
      const c = constantMap.get(tensorName);
      return c ? c.rawData : null;
    },
    constantIntValues(tensorName: string): number[] | null {
      const c = constantMap.get(tensorName);
      if (!c || !c.rawData || c.rawData.byteLength === 0) return null;
      // Copy to aligned buffer to avoid typed array alignment issues
      const aligned = new ArrayBuffer(c.rawData.byteLength);
      new Uint8Array(aligned).set(c.rawData);
      if (c.dataType === 'int64') {
        const view = new BigInt64Array(aligned);
        return Array.from(view, (v) => Number(v));
      } else if (c.dataType === 'int32') {
        return Array.from(new Int32Array(aligned));
      } else if (c.dataType === 'uint32') {
        return Array.from(new Uint32Array(aligned));
      }
      return null;
    },
    tensorShape(tensorName: string): (number | string)[] | null {
      return graph.shapes?.get(tensorName) ?? null;
    },
    tensorDataType(tensorName: string): string | null {
      return graph.dataTypes?.get(tensorName) ?? null;
    },
    findProducerNode(tensorName: string): NodeIR | null {
      return graph.nodes.find((n) => n.outputs.includes(tensorName)) ?? null;
    },
    markDead(tensorName: string): void {
      deadTensors.add(tensorName);
    },
    isDead(tensorName: string): boolean {
      return deadTensors.has(tensorName);
    },
  };

  // --- Emit graph inputs ---
  emit('// Graph inputs');
  for (const input of graph.inputs) {
    const varName = getOrDeclare(input.name);
    const shape = JSON.stringify(input.shape);
    emit(
      `const ${varName} = builder.input('${input.name}', { dataType: '${input.dataType}', shape: ${shape} });`,
    );
  }
  emit('');

  // --- Emit constants (from weight file) ---
  if (graph.constants.length > 0) {
    emit('// Constants (loaded from weights file)');
    for (const c of graph.constants) {
      // Skip empty tensors (e.g. ONNX Resize unused roi/scales with shape [0])
      if (c.shape.some((d) => d === 0)) continue;
      const varName = getOrDeclare(c.name);
      const shape = JSON.stringify(c.shape.map((d) => (typeof d === 'number' ? d : 0)));
      const typedArray = getTypedArrayName(c.dataType);
      emit('{');
      indentLevel++;
      emit(`const sl = weights.getSlice('${c.name}');`);
      emit(`const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);`);
      emit(
        `${varName} = builder.constant({ dataType: '${c.dataType}', shape: ${shape} }, new ${typedArray}(buf));`,
      );
      indentLevel--;
      emit('}');
    }
    emit('');

    // Re-declare constants with let at the top
    // Actually, we need to handle scoping — use let declarations before the blocks
    // Let me restructure: declare all constants with let before the blocks
  }

  // --- Emit nodes ---
  emit('// Graph operations');
  for (const node of graph.nodes) {
    // Propagate dead state: if any input is dead, skip this op entirely
    const hasDead = node.inputs.some(
      (name) => name !== '' && emitterImpl.isDead(name),
    );
    if (hasDead) {
      emit(`// SKIPPED: ${node.opType} — depends on unsupported op output`);
      for (const out of node.outputs) {
        if (out !== '') {
          emitterImpl.markDead(out);
          const varName = getOrDeclare(out);
          emit(`const ${varName} = undefined; // dead — upstream unsupported`);
        }
      }
      continue;
    }

    const opEmitter = getEmitter(graph.format, node.opType);
    if (opEmitter) {
      emit(`// ${node.opType}`);
      opEmitter(node, emitterImpl);
    } else {
      emit(`// UNSUPPORTED: ${node.opType} — no WebNN equivalent`);
      for (const out of node.outputs) {
        if (out !== '') {
          emitterImpl.markDead(out);
          const varName = getOrDeclare(out);
          emit(`const ${varName} = undefined; // unsupported op`);
        }
      }
    }
  }
  emit('');

  // --- Build named outputs ---
  emit('// Build graph');
  emit('const namedOutputs = {};');
  const liveOutputs = graph.outputs.filter((o) => !emitterImpl.isDead(o.name));
  if (liveOutputs.length > 0) {
    for (const output of liveOutputs) {
      const varName = getOrDeclare(output.name);
      emit(`namedOutputs['${output.name}'] = ${varName};`);
    }
  } else {
    // All original outputs depend on unsupported ops — export frontier tensors instead
    const frontier = findFrontierTensors(graph, emitterImpl);
    emit('// NOTE: All original outputs depend on unsupported ops (e.g. TopK, Range, Mod).');
    emit('// Exporting the last computed tensors before the unsupported section.');
    for (const name of frontier) {
      const varName = getOrDeclare(name);
      emit(`namedOutputs['${name}'] = ${varName};`);
    }
  }
  emit('// Capture actual output operand shapes (may differ from metadata for dynamic models)');  
  emit('const outputShapes = {};');
  emit('for (const [name, operand] of Object.entries(namedOutputs)) {');
  indentLevel++;
  emit('outputShapes[name] = Array.from(operand.shape);');
  indentLevel--;
  emit('}');
  emit('return { graph: await builder.build(namedOutputs), outputShapes };');
  indentLevel--;
  emit('}');
  emit('');

  // --- Main runner function ---
  emit('/**');
  emit(' * Run inference');
  emit(` * @param {string} deviceType - 'cpu', 'gpu', or 'npu' (default: 'cpu')`);
  emit(' */');
  emit(`async function main(deviceType = 'cpu') {`);
  indentLevel++;
  emit('if (!navigator.ml) {');
  indentLevel++;
  emit("throw new Error('WebNN is not supported in this browser.');");
  indentLevel--;
  emit('}');
  emit('');
  emit(`const context = await navigator.ml.createContext({ deviceType });`);
  emit(
    `const weights = await WeightsFile.load('${weightsFileName}', '${manifestFileName}');`,
  );
  emit('const buildStart = performance.now();');
  emit('const graph = await buildGraph(context, weights);');
  emit("console.log(`Graph build: ${(performance.now() - buildStart).toFixed(2)}ms on ${deviceType.toUpperCase()}`)");
  emit('');
  emit('// Create input tensors');
  for (const input of graph.inputs) {
    const typedArray = getTypedArrayName(input.dataType);
    const numericShape = input.shape.map((d) => (typeof d === 'number' ? d : 1));
    const totalSize = numericShape.reduce((a, b) => a * b, 1);
    emit(
      `const inputData_${toJsVarName(input.name)} = new ${typedArray}(${totalSize}); // shape: ${JSON.stringify(input.shape)}`,
    );
    emit(
      `const inputTensor_${toJsVarName(input.name)} = await context.createTensor({ dataType: '${input.dataType}', shape: ${JSON.stringify(numericShape)}, writable: true });`,
    );
    emit(
      `context.writeTensor(inputTensor_${toJsVarName(input.name)}, inputData_${toJsVarName(input.name)});`,
    );
  }
  emit('');
  const effectiveOutputs = computeEffectiveOutputs(graph);
  const effectiveOutputTypes = computeEffectiveOutputTypes(graph);
  emit('// Create output tensors — use actual shapes from built graph');
  for (const output of effectiveOutputs) {
    const dt = effectiveOutputTypes.get(output.name) ?? output.dataType;
    emit(
      `const outputTensor_${toJsVarName(output.name)} = await context.createTensor({ dataType: '${dt}', shape: graph.outputShapes['${escapeSingleQuotes(output.name)}'], readable: true });`,
    );
  }
  emit('');
  emit('const inputs = {};');
  for (const input of graph.inputs) {
    emit(`inputs['${input.name}'] = inputTensor_${toJsVarName(input.name)};`);
  }
  emit('const outputs = {};');
  for (const output of effectiveOutputs) {
    emit(`outputs['${escapeSingleQuotes(output.name)}'] = outputTensor_${toJsVarName(output.name)};`);
  }
  emit('');
  emit('const start = performance.now();');
  emit('context.dispatch(graph.graph, inputs, outputs);');
  emit('');
  emit('// Read results');
  for (const output of effectiveOutputs) {
    const dt = effectiveOutputTypes.get(output.name) ?? output.dataType;
    const typedArray = getTypedArrayName(dt);
    emit(`const result_${toJsVarName(output.name)} = new ${typedArray}(await context.readTensor(outputTensor_${toJsVarName(output.name)}));`);
  }
  emit("const elapsed = (performance.now() - start).toFixed(2);");
  emit('console.log(`Inference: ${elapsed}ms (1 run) on ${deviceType.toUpperCase()}`);');
  emit('');
  emit('// Benchmark: 50 runs');
  emit('const NUM_RUNS = 50;');
  emit('const runTimes = [];');
  emit('for (let i = 0; i < NUM_RUNS; i++) {');
  indentLevel++;
  emit('const t0 = performance.now();');
  emit('context.dispatch(graph.graph, inputs, outputs);');
  for (const output of effectiveOutputs) {
    const dt = effectiveOutputTypes.get(output.name) ?? output.dataType;
    const typedArray = getTypedArrayName(dt);
    emit(`new ${typedArray}(await context.readTensor(outputTensor_${toJsVarName(output.name)}));`);
  }
  emit('runTimes.push(performance.now() - t0);');
  indentLevel--;
  emit('}');
  emit('const avgTime = (runTimes.reduce((a, b) => a + b, 0) / NUM_RUNS).toFixed(2);');
  emit('const sorted = [...runTimes].sort((a, b) => a - b);');
  emit('const medianTime = (NUM_RUNS % 2 ? sorted[NUM_RUNS >> 1] : (sorted[NUM_RUNS / 2 - 1] + sorted[NUM_RUNS / 2]) / 2).toFixed(2);');
  emit("console.log(`Inference: ${avgTime}ms (average \\u00b7 ${NUM_RUNS} runs) on ${deviceType.toUpperCase()}`)");
  emit("console.log(`Inference: ${medianTime}ms (median \\u00b7 ${NUM_RUNS} runs) on ${deviceType.toUpperCase()}`)");
  emit('');
  emit('// Access results');
  for (const output of effectiveOutputs) {
    emit(`console.log('${escapeSingleQuotes(output.name)}:', result_${toJsVarName(output.name)});`);
  }
  emit('');
  const returnEntries = effectiveOutputs.map(
    (o) => `'${escapeSingleQuotes(o.name)}': result_${toJsVarName(o.name)}`
  );
  emit(`return { ${returnEntries.join(', ')} };`);
  indentLevel--;
  emit('}');

  return lines.join('\n') + '\n';
}

// Fix the constant scoping issue — constants need to be declared with let before block assignments
export function generateJavaScriptFixed(
  graph: GraphIR,
  options: GenerateJsOptions = {},
): string {
  const {
    weightsFileName = 'model.weights',
    manifestFileName = 'model.manifest.json',
    includeWeightsLoader = true,
  } = options;

  const lines: string[] = [];
  const indent = '  ';
  let indentLevel = 0;

  function emit(line: string): void {
    if (line === '') {
      lines.push('');
    } else {
      lines.push(indent.repeat(indentLevel) + line);
    }
  }

  // --- WeightsFile helper class ---
  if (includeWeightsLoader) {
    emitWeightsFileClass(emit, () => indentLevel++, () => indentLevel--);
    emit('');
  }

  // --- buildGraph function ---
  emit('/**');
  emit(` * Build WebNN graph for model: ${graph.name}`);
  emit(` * Source format: ${graph.format}`);
  emit(' */');
  emit('async function buildGraph(context, weights) {');
  indentLevel++;
  emit('const builder = new MLGraphBuilder(context);');
  emit('');

  // Track variables
  const declaredVars = new Set<string>();
  const constantNames = new Set<string>();
  const constantMap = new Map<string, ConstantInfo>();
  const varMap = new Map<string, string>();
  const deadTensors = new Set<string>();

  for (const c of graph.constants) {
    constantNames.add(c.name);
    constantMap.set(c.name, c);
  }

  function makeVarName(tensorName: string): string {
    let name = toJsVarName(tensorName);
    if (declaredVars.has(name)) {
      let i = 2;
      while (declaredVars.has(`${name}_${i}`)) i++;
      name = `${name}_${i}`;
    }
    return name;
  }

  function getOrDeclare(tensorName: string): string {
    if (varMap.has(tensorName)) return varMap.get(tensorName)!;
    const name = makeVarName(tensorName);
    declaredVars.add(name);
    varMap.set(tensorName, name);
    return name;
  }

  const emitterImpl: CodeEmitter = {
    ref: (t: string) => getOrDeclare(t),
    declare: (t: string) => getOrDeclare(t),
    line: (code: string) => emit(code),
    comment: (text: string) => emit(`// ${text}`),
    isConstant: (t: string) => constantNames.has(t),
    constantShape: (t: string) => {
      const c = constantMap.get(t);
      return c ? c.shape.map((d) => (typeof d === 'number' ? d : 0)) : [];
    },
    constantDataType: (t: string) => constantMap.get(t)?.dataType ?? 'float32',
    constantRawData: (t: string) => constantMap.get(t)?.rawData ?? null,
    constantIntValues(tensorName: string): number[] | null {
      const c = constantMap.get(tensorName);
      if (!c || !c.rawData || c.rawData.byteLength === 0) return null;
      const aligned = new ArrayBuffer(c.rawData.byteLength);
      new Uint8Array(aligned).set(c.rawData);
      if (c.dataType === 'int64') {
        const view = new BigInt64Array(aligned);
        return Array.from(view, (v) => Number(v));
      } else if (c.dataType === 'int32') {
        return Array.from(new Int32Array(aligned));
      } else if (c.dataType === 'uint32') {
        return Array.from(new Uint32Array(aligned));
      }
      return null;
    },
    tensorShape: (t: string) => graph.shapes?.get(t) ?? null,
    tensorDataType: (t: string) => graph.dataTypes?.get(t) ?? null,
    findProducerNode: (t: string) => graph.nodes.find((n) => n.outputs.includes(t)) ?? null,
    markDead: (t: string) => { deadTensors.add(t); },
    isDead: (t: string) => deadTensors.has(t),
  };

  // --- Emit graph inputs ---
  emit('// Graph inputs');
  for (const input of graph.inputs) {
    const varName = getOrDeclare(input.name);
    const shape = JSON.stringify(input.shape);
    emit(`const ${varName} = builder.input('${input.name}', { dataType: '${input.dataType}', shape: ${shape} });`);
  }
  emit('');

  // --- Emit constants ---
  if (graph.constants.length > 0) {
    emit('// Constants (loaded from weights file)');
    for (const c of graph.constants) {
      // Skip empty tensors (e.g. ONNX Resize unused roi/scales with shape [0])
      if (c.shape.some((d) => d === 0)) continue;
      const varName = getOrDeclare(c.name);
      const shape = JSON.stringify(c.shape.map((d) => (typeof d === 'number' ? d : 0)));
      const typedArray = getTypedArrayName(c.dataType);
      emit(`const ${varName} = (() => {`);
      indentLevel++;
      emit(`const sl = weights.getSlice('${escapeSingleQuotes(c.name)}');`);
      emit(`const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);`);
      emit(`return builder.constant({ dataType: '${c.dataType}', shape: ${shape} }, new ${typedArray}(buf));`);
      indentLevel--;
      emit(`})();`);
    }
    emit('');
  }

  // --- Emit nodes ---
  emit('// Graph operations');
  for (const node of graph.nodes) {
    // Propagate dead state: if any input is dead, skip this op entirely
    const hasDead = node.inputs.some(
      (name) => name !== '' && emitterImpl.isDead(name),
    );
    if (hasDead) {
      emit(`// SKIPPED: ${node.opType} — depends on unsupported op output`);
      for (const out of node.outputs) {
        if (out !== '') {
          emitterImpl.markDead(out);
          const varName = getOrDeclare(out);
          emit(`const ${varName} = undefined; // dead — upstream unsupported`);
        }
      }
      continue;
    }

    const opEmitter = getEmitter(graph.format, node.opType);
    if (opEmitter) {
      emit(`// ${node.opType}`);
      opEmitter(node, emitterImpl);
    } else {
      emit(`// UNSUPPORTED: ${node.opType} — no WebNN equivalent`);
      for (const out of node.outputs) {
        if (out !== '') {
          emitterImpl.markDead(out);
          const varName = getOrDeclare(out);
          emit(`const ${varName} = undefined; // unsupported op`);
        }
      }
    }
  }
  emit('');

  // --- Build graph ---
  emit('// Build graph');
  emit('const namedOutputs = {};');
  const liveOutputs = graph.outputs.filter((o) => !emitterImpl.isDead(o.name));
  if (liveOutputs.length > 0) {
    for (const output of liveOutputs) {
      const varName = getOrDeclare(output.name);
      emit(`namedOutputs['${escapeSingleQuotes(output.name)}'] = ${varName};`);
    }
  } else {
    // All original outputs depend on unsupported ops — export frontier tensors instead
    const frontier = findFrontierTensors(graph, emitterImpl);
    emit('// NOTE: All original outputs depend on unsupported ops (e.g. TopK, Range, Mod).');
    emit('// Exporting the last computed tensors before the unsupported section.');
    for (const name of frontier) {
      const varName = getOrDeclare(name);
      emit(`namedOutputs['${escapeSingleQuotes(name)}'] = ${varName};`);
    }
  }
  emit('// Capture actual output operand shapes (may differ from metadata for dynamic models)');
  emit('const outputShapes = {};');
  emit('for (const [name, operand] of Object.entries(namedOutputs)) {');
  indentLevel++;
  emit('outputShapes[name] = Array.from(operand.shape);');
  indentLevel--;
  emit('}');
  emit('return { graph: await builder.build(namedOutputs), outputShapes };');
  indentLevel--;
  emit('}');
  emit('');

  // --- Main runner ---
  emitMainFunction(emit, graph, weightsFileName, manifestFileName, () => indentLevel++, () => indentLevel--);

  return lines.join('\n') + '\n';
}

function emitWeightsFileClass(
  emit: (s: string) => void,
  inc: () => void,
  dec: () => void,
): void {
  emit('class WeightsFile {');
  inc();
  emit('constructor(buffer, manifest) {');
  inc();
  emit('this.buffer = buffer;');
  emit('this.manifest = manifest;');
  dec();
  emit('}');
  emit('');
  emit('static async load(weightsPath, manifestPath) {');
  inc();
  emit('const [wRes, mRes] = await Promise.all([fetch(weightsPath), fetch(manifestPath)]);');
  emit("if (!wRes.ok) throw new Error('Failed to load weights: ' + wRes.statusText);");
  emit("if (!mRes.ok) throw new Error('Failed to load manifest: ' + mRes.statusText);");
  emit('const buffer = await wRes.arrayBuffer();');
  emit('const manifest = await mRes.json();');
  emit("if (manifest.format !== 'wg-weights-manifest') throw new Error('Invalid manifest format');");
  emit("const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));");
  emit("if (magic !== 'WGWT') throw new Error('Invalid weights file');");
  emit('return new WeightsFile(buffer, manifest);');
  dec();
  emit('}');
  emit('');
  emit('getSlice(name) {');
  inc();
  emit('const t = this.manifest.tensors[name];');
  emit("if (!t) throw new Error('Tensor not found: ' + name);");
  emit('return t;');
  dec();
  emit('}');
  emit('');
  emit('getData(name) {');
  inc();
  emit('const t = this.getSlice(name);');
  emit('return this.buffer.slice(t.byteOffset, t.byteOffset + t.byteLength);');
  dec();
  emit('}');
  dec();
  emit('}');
}

function emitMainFunction(
  emit: (s: string) => void,
  graph: GraphIR,
  weightsFileName: string,
  manifestFileName: string,
  inc: () => void,
  dec: () => void,
): void {
  emit("async function main(deviceType = 'cpu') {");
  inc();
  emit("if (!navigator.ml) throw new Error('WebNN is not supported in this browser.');");
  emit('const context = await navigator.ml.createContext({ deviceType });');
  emit(`const weights = await WeightsFile.load('${weightsFileName}', '${manifestFileName}');`);
  emit('const buildStart = performance.now();');
  emit('const graph = await buildGraph(context, weights);');
  emit('console.log(`Graph build: ${(performance.now() - buildStart).toFixed(2)}ms on ${deviceType.toUpperCase()}`)');
  emit('');
  emit('// Create input tensors');
  for (const input of graph.inputs) {
    const typedArray = getTypedArrayName(input.dataType);
    const numericShape = input.shape.map((d) => (typeof d === 'number' ? d : 1));
    const totalSize = numericShape.reduce((a, b) => a * b, 1);
    const vn = toJsVarName(input.name);
    emit(`const inputData_${vn} = new ${typedArray}(${totalSize}); // ${JSON.stringify(input.shape)}`);
    emit(`const inputTensor_${vn} = await context.createTensor({ dataType: '${input.dataType}', shape: ${JSON.stringify(numericShape)}, writable: true });`);
    emit(`context.writeTensor(inputTensor_${vn}, inputData_${vn});`);
  }
  emit('');
  const effectiveOutputs = computeEffectiveOutputs(graph);
  const effectiveOutputTypes = computeEffectiveOutputTypes(graph);
  emit('// Create output tensors — use actual shapes from built graph');
  for (const output of effectiveOutputs) {
    const vn = toJsVarName(output.name);
    const dt = effectiveOutputTypes.get(output.name) ?? output.dataType;
    emit(`const outputTensor_${vn} = await context.createTensor({ dataType: '${dt}', shape: graph.outputShapes['${escapeSingleQuotes(output.name)}'], readable: true });`);
  }
  emit('');
  emit('const inputs = {');
  inc();
  for (const input of graph.inputs) {
    emit(`'${escapeSingleQuotes(input.name)}': inputTensor_${toJsVarName(input.name)},`);
  }
  dec();
  emit('};');
  emit('const outputs = {');
  inc();
  for (const output of effectiveOutputs) {
    emit(`'${escapeSingleQuotes(output.name)}': outputTensor_${toJsVarName(output.name)},`);
  }
  dec();
  emit('};');
  emit('');
  emit('const start = performance.now();');
  emit('context.dispatch(graph.graph, inputs, outputs);');
  emit('');
  emit('// Read results');
  for (const output of effectiveOutputs) {
    const dt = effectiveOutputTypes.get(output.name) ?? output.dataType;
    const typedArray = getTypedArrayName(dt);
    const vn = toJsVarName(output.name);
    emit(`const result_${vn} = new ${typedArray}(await context.readTensor(outputTensor_${vn}));`);
  }
  emit("console.log(`Inference: ${(performance.now() - start).toFixed(2)}ms (1 run) on ${deviceType.toUpperCase()}`)");
  emit('');
  emit('// Benchmark: 50 runs');
  emit('const NUM_RUNS = 50;');
  emit('const runTimes = [];');
  emit('for (let i = 0; i < NUM_RUNS; i++) {');
  inc();
  emit('const t0 = performance.now();');
  emit('context.dispatch(graph.graph, inputs, outputs);');
  for (const output of effectiveOutputs) {
    const dt = effectiveOutputTypes.get(output.name) ?? output.dataType;
    const typedArray = getTypedArrayName(dt);
    const vn = toJsVarName(output.name);
    emit(`new ${typedArray}(await context.readTensor(outputTensor_${vn}));`);
  }
  emit('runTimes.push(performance.now() - t0);');
  dec();
  emit('}');
  emit('const avgTime = (runTimes.reduce((a, b) => a + b, 0) / NUM_RUNS).toFixed(2);');
  emit('const sorted = [...runTimes].sort((a, b) => a - b);');
  emit('const medianTime = (NUM_RUNS % 2 ? sorted[NUM_RUNS >> 1] : (sorted[NUM_RUNS / 2 - 1] + sorted[NUM_RUNS / 2]) / 2).toFixed(2);');
  emit("console.log(`Inference: ${avgTime}ms (average \\u00b7 ${NUM_RUNS} runs) on ${deviceType.toUpperCase()}`)");
  emit("console.log(`Inference: ${medianTime}ms (median \\u00b7 ${NUM_RUNS} runs) on ${deviceType.toUpperCase()}`)");
  emit('');
  const returnEntries = effectiveOutputs.map(
    (o) => `'${escapeSingleQuotes(o.name)}': result_${toJsVarName(o.name)}`
  );
  emit(`return { ${returnEntries.join(', ')} };`);
  dec();
  emit('}');
}

function escapeSingleQuotes(s: string): string {
  return s.replace(/'/g, "\\'");
}

/**
 * Compute dead tensors exactly as generateJavaScriptFixed does — including
 * conditional markDead() calls made by emitters (e.g. Shape with unknown input shape).
 *
 * Runs each emitter through a no-op mock emitter so that emitters which call
 * emitter.markDead() (rather than just failing to register) are correctly handled.
 */
function computeDeadTensors(graph: GraphIR): Set<string> {
  const constantNames = new Set(graph.constants.map((c) => c.name));
  const constantMap = new Map(graph.constants.map((c) => [c.name, c]));
  const dead = new Set<string>();
  const varMap = new Map<string, string>();

  function ref(t: string): string { return varMap.get(t) ?? toJsVarName(t); }

  const mockEmitter: CodeEmitter = {
    ref,
    declare: (t: string) => { const v = toJsVarName(t); varMap.set(t, v); return v; },
    line: () => { /* no-op */ },
    comment: () => { /* no-op */ },
    isConstant: (t: string) => constantNames.has(t),
    constantShape: (t: string) => {
      const c = constantMap.get(t);
      return c ? c.shape.map((d) => (typeof d === 'number' ? d : 0)) : [];
    },
    constantDataType: (t: string) => constantMap.get(t)?.dataType ?? 'float32',
    constantRawData: (t: string) => constantMap.get(t)?.rawData ?? null,
    constantIntValues: (t: string) => {
      const c = constantMap.get(t);
      if (!c || !c.rawData || c.rawData.byteLength === 0) return null;
      const aligned = new ArrayBuffer(c.rawData.byteLength);
      new Uint8Array(aligned).set(c.rawData);
      if (c.dataType === 'int64') return Array.from(new BigInt64Array(aligned), (v) => Number(v));
      if (c.dataType === 'int32') return Array.from(new Int32Array(aligned));
      if (c.dataType === 'uint32') return Array.from(new Uint32Array(aligned));
      return null;
    },
    tensorShape: (t: string) => graph.shapes?.get(t) ?? null,
    tensorDataType: (t: string) => graph.dataTypes?.get(t) ?? null,
    findProducerNode: (t: string) => graph.nodes.find((n) => n.outputs.includes(t)) ?? null,
    markDead: (t: string) => { dead.add(t); },
    isDead: (t: string) => dead.has(t),
  };

  for (const node of graph.nodes) {
    // Propagate dead state: if any input is dead, mark all outputs dead
    const hasDead = node.inputs.some((name) => name !== '' && dead.has(name));
    if (hasDead) {
      for (const out of node.outputs) {
        if (out !== '') dead.add(out);
      }
      continue;
    }

    const opEmitter = getEmitter(graph.format, node.opType);
    if (opEmitter) {
      // Run emitter so it can call emitter.markDead() if needed (e.g. Shape with unknown shape)
      try { opEmitter(node, mockEmitter); } catch { /* ignore codegen errors in simulation */ }
      // Any outputs not declared (not in varMap) and explicitly marked dead stay dead
    } else {
      for (const out of node.outputs) {
        if (out !== '') dead.add(out);
      }
    }
  }

  return dead;
}

/**
 * Compute the actual graph outputs that buildGraph() will produce.
 *
 * When all original outputs depend on unsupported ops (e.g. TopK, Range, Mod),
 * the codegen substitutes "frontier" tensors — the last computed values before
 * the dead zone. This function replicates the dead-propagation logic to return
 * the same set of TensorInfo objects that the generated namedOutputs will use.
 */
export function computeEffectiveOutputs(graph: GraphIR): TensorInfo[] {
  const constantNames = new Set(graph.constants.map((c) => c.name));
  const graphInputNames = new Set(graph.inputs.map((i) => i.name));
  const dead = computeDeadTensors(graph);

  // Check if any original outputs are live
  const liveOutputs = graph.outputs.filter((o) => !dead.has(o.name));
  if (liveOutputs.length > 0) return liveOutputs;

  // All original outputs dead — find frontier tensors (same logic as codegen)
  const frontier: string[] = [];
  for (const node of graph.nodes) {
    const anyOutputDead = node.outputs.some((o) => o !== '' && dead.has(o));
    if (!anyOutputDead) continue;
    for (const inp of node.inputs) {
      if (inp === '' || dead.has(inp)) continue;
      if (constantNames.has(inp) || graphInputNames.has(inp)) continue;
      frontier.push(inp);
    }
  }

  // Deduplicate while preserving order
  const seen = new Set<string>();
  const unique: string[] = [];
  for (const name of frontier) {
    if (!seen.has(name)) { seen.add(name); unique.push(name); }
  }

  // Build TensorInfo for each frontier tensor from the graph's shape/dataType maps
  return unique.map((name) => ({
    name,
    dataType: graph.dataTypes?.get(name) ?? 'float32',
    shape: graph.shapes?.get(name) ?? [],
  }));
}

/**
 * Compute effective output data types for the test harness.
 *
 * TFLite quantized models store output tensors as int8/uint8, but the generated
 * WebNN graph dequantizes all computations to float32. The actual MLGraph
 * outputs are therefore float32, not the original quantized type.
 *
 * We trace backward from each output: if any ancestor op dequantizes its inputs
 * (i.e. the model format is tflite and the type is quantized), the effective
 * output type is float32.
 */
export function computeEffectiveOutputTypes(graph: GraphIR): Map<string, MLOperandDataType> {
  const effectiveTypes = new Map<string, MLOperandDataType>();

  if (graph.format !== 'tflite') {
    // ONNX models don't do implicit dequantization in our pipeline
    for (const o of graph.outputs) effectiveTypes.set(o.name, o.dataType);
    return effectiveTypes;
  }

  // Build a producer map: tensor name → producing NodeIR
  const producerMap = new Map<string, NodeIR>();
  for (const node of graph.nodes) {
    for (const out of node.outputs) {
      producerMap.set(out, node);
    }
  }

  // Ops that explicitly re-quantize (output stays int8/uint8)
  const quantizeOps = new Set(['QUANTIZE']);

  // Walk backward from each output to determine effective type
  for (const output of graph.outputs) {
    if (output.dataType !== 'int8' && output.dataType !== 'uint8') {
      effectiveTypes.set(output.name, output.dataType);
      continue;
    }

    // Check if the producing chain dequantizes
    let tensorName = output.name;
    let effectiveType: MLOperandDataType = output.dataType;
    const visited = new Set<string>();

    while (tensorName && !visited.has(tensorName)) {
      visited.add(tensorName);
      const producer = producerMap.get(tensorName);
      if (!producer) break;

      if (quantizeOps.has(producer.opType)) {
        // Explicit quantize — output stays quantized
        effectiveType = output.dataType;
        break;
      }

      // Shape-preserving pass-through ops: check their first input
      const passThroughOps = new Set([
        'RESHAPE', 'TRANSPOSE', 'SQUEEZE', 'EXPAND_DIMS',
        'PACK', 'UNPACK', 'SPLIT', 'SPLIT_V',
        'CONCATENATION', 'GATHER', 'SLICE', 'STRIDED_SLICE',
        'PAD', 'PADV2', 'TILE', 'MIRROR_PAD',
      ]);

      if (passThroughOps.has(producer.opType)) {
        // These ops preserve the data type of their input — trace deeper
        tensorName = producer.inputs[0];
        continue;
      }

      // Computational ops (FULLY_CONNECTED, CONV_2D, matmul, etc.)
      // — these call emitDequantizeIfNeeded and produce float32
      effectiveType = 'float32';
      break;
    }

    effectiveTypes.set(output.name, effectiveType);
  }

  return effectiveTypes;
}
