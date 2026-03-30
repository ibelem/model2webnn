// JavaScript code generator — emits .js with MLGraphBuilder calls
// Produces a self-contained buildGraph function + WeightsFile helper.

import type { GraphIR, ConstantInfo, NodeIR } from '../ir/graph.js';
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
    const opEmitter = getEmitter(graph.format, node.opType);
    if (opEmitter) {
      emit(`// ${node.opType}`);
      opEmitter(node, emitterImpl);
    } else {
      emit(`// UNSUPPORTED: ${node.opType} — skipped`);
      // Still declare outputs to avoid reference errors
      for (const out of node.outputs) {
        if (out !== '') {
          const varName = getOrDeclare(out);
          emit(`const ${varName} = undefined; // TODO: implement ${node.opType}`);
        }
      }
    }
  }
  emit('');

  // --- Build named outputs ---
  emit('// Build graph');
  emit('const namedOutputs = {};');
  for (const output of graph.outputs) {
    const varName = getOrDeclare(output.name);
    emit(`namedOutputs['${output.name}'] = ${varName};`);
  }
  emit('return await builder.build(namedOutputs);');
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
  emit('const graph = await buildGraph(context, weights);');
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
  emit('// Create output tensors');
  for (const output of graph.outputs) {
    const numericShape = output.shape.map((d) => (typeof d === 'number' ? d : 1));
    emit(
      `const outputTensor_${toJsVarName(output.name)} = await context.createTensor({ dataType: '${output.dataType}', shape: ${JSON.stringify(numericShape)}, readable: true });`,
    );
  }
  emit('');
  emit('const inputs = {};');
  for (const input of graph.inputs) {
    emit(`inputs['${input.name}'] = inputTensor_${toJsVarName(input.name)};`);
  }
  emit('const outputs = {};');
  for (const output of graph.outputs) {
    emit(`outputs['${output.name}'] = outputTensor_${toJsVarName(output.name)};`);
  }
  emit('');
  emit('const start = performance.now();');
  emit('context.dispatch(graph, inputs, outputs);');
  emit('');
  emit('// Read results');
  for (const output of graph.outputs) {
    const typedArray = getTypedArrayName(output.dataType);
    emit(`const result_${toJsVarName(output.name)} = new ${typedArray}(await context.readTensor(outputTensor_${toJsVarName(output.name)}));`);
  }
  emit("const elapsed = (performance.now() - start).toFixed(2);");
  emit('console.log(`Inference completed in ${elapsed}ms`);');
  emit('');
  emit('// Access results');
  for (const output of graph.outputs) {
    emit(`console.log('${output.name}:', result_${toJsVarName(output.name)});`);
  }
  emit('');
  const returnEntries = graph.outputs.map(
    (o) => `'${o.name}': result_${toJsVarName(o.name)}`
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
    const opEmitter = getEmitter(graph.format, node.opType);
    if (opEmitter) {
      emit(`// ${node.opType}`);
      opEmitter(node, emitterImpl);
    } else {
      emit(`// UNSUPPORTED: ${node.opType}`);
      for (const out of node.outputs) {
        if (out !== '') {
          const varName = getOrDeclare(out);
          emit(`const ${varName} = undefined; // TODO: implement ${node.opType}`);
        }
      }
    }
  }
  emit('');

  // --- Build graph ---
  emit('// Build graph');
  emit('const namedOutputs = {};');
  for (const output of graph.outputs) {
    const varName = getOrDeclare(output.name);
    emit(`namedOutputs['${escapeSingleQuotes(output.name)}'] = ${varName};`);
  }
  emit('return await builder.build(namedOutputs);');
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
  emit('const graph = await buildGraph(context, weights);');
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
  emit('// Create output tensors');
  for (const output of graph.outputs) {
    const numericShape = output.shape.map((d) => (typeof d === 'number' ? d : 1));
    const vn = toJsVarName(output.name);
    emit(`const outputTensor_${vn} = await context.createTensor({ dataType: '${output.dataType}', shape: ${JSON.stringify(numericShape)}, readable: true });`);
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
  for (const output of graph.outputs) {
    emit(`'${escapeSingleQuotes(output.name)}': outputTensor_${toJsVarName(output.name)},`);
  }
  dec();
  emit('};');
  emit('');
  emit('const start = performance.now();');
  emit('context.dispatch(graph, inputs, outputs);');
  emit('');
  emit('// Read results');
  for (const output of graph.outputs) {
    const typedArray = getTypedArrayName(output.dataType);
    const vn = toJsVarName(output.name);
    emit(`const result_${vn} = new ${typedArray}(await context.readTensor(outputTensor_${vn}));`);
  }
  emit("console.log(`Inference completed in ${(performance.now() - start).toFixed(2)}ms`);");
  const returnEntries = graph.outputs.map(
    (o) => `'${escapeSingleQuotes(o.name)}': result_${toJsVarName(o.name)}`
  );
  emit(`return { ${returnEntries.join(', ')} };`);
  dec();
  emit('}');
}

function escapeSingleQuotes(s: string): string {
  return s.replace(/'/g, "\\'");
}
