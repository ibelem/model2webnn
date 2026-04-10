// HTML code generator — emits a self-contained runnable .html file
// Embeds the generated JS code + WebNN test harness with device selector.

import type { GraphIR } from '../ir/graph.js';
import { getTypedArrayName } from '../ir/graph.js';
import { generateJavaScriptFixed, computeEffectiveOutputTypes, computeEffectiveOutputs } from './javascript.js';

export interface GenerateHtmlOptions {
  weightsFileName?: string;
  manifestFileName?: string;
  title?: string;
}

export function generateHtml(
  graph: GraphIR,
  options: GenerateHtmlOptions = {},
): string {
  const {
    weightsFileName = 'model.weights',
    manifestFileName = 'model.manifest.json',
    title = `WebNN · ${graph.name}`,
  } = options;

  const jsCode = generateJavaScriptFixed(graph, {
    weightsFileName,
    manifestFileName,
    includeWeightsLoader: true,
  });

  // Build input info for the harness
  const inputInfoLines: string[] = [];
  for (const input of graph.inputs) {
    const typedArray = getTypedArrayName(input.dataType);
    const numericShape = input.shape.map((d) => (typeof d === 'number' ? d : 1));
    const totalSize = numericShape.reduce((a, b) => a * b, 1);
    inputInfoLines.push(
      `    { name: '${escapeSingleQuotes(input.name)}', dataType: '${input.dataType}', shape: ${JSON.stringify(input.shape)}, TypedArray: ${typedArray}, size: ${totalSize} },`,
    );
  }

  const effectiveOutputs = computeEffectiveOutputs(graph);
  const effectiveOutputTypes = computeEffectiveOutputTypes(graph);
  const outputInfoLines: string[] = [];
  for (const output of effectiveOutputs) {
    const dt = effectiveOutputTypes.get(output.name) ?? output.dataType;
    const typedArray = getTypedArrayName(dt);
    const numericShape = output.shape.map((d) => (typeof d === 'number' ? d : 1));
    const totalSize = numericShape.reduce((a, b) => a * b, 1);
    outputInfoLines.push(
      `    { name: '${escapeSingleQuotes(output.name)}', dataType: '${dt}', shape: ${JSON.stringify(output.shape)}, TypedArray: ${typedArray}, size: ${totalSize} },`,
    );
  }

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(title)}</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: rgba(0, 47, 167, 0.02); color: #1a1a1a; padding: 1rem; }
    .container { margin: 0 auto; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    .subtitle { color: #666; margin-bottom: 1rem; font-size: 0.9rem; }
    .card { background: #fff; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .model-info { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; }
    @media (max-width: 600px) { .model-info { grid-template-columns: 1fr; } }
    .controls { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
    label { font-weight: 600; font-size: 0.9rem; }
    select { padding: 0.16rem 0.5rem; border: 1px solid #ddd; font-size: 0.9rem; }
    button { background: rgba(0, 47, 167, 1); color: #fff; border: none; padding: 0.25rem 1rem; font-size: 0.9rem; cursor: pointer; }
    button:hover { background: rgba(0, 47, 167, 1); }
    button:disabled { background: #ccc; cursor: not-allowed; }
    .status { padding: 1rem; margin-top: 1rem; font-size: 0.9rem; }
    .status.info { background: #e3f2fd; border-left: 3px solid #2196F3; }
    .status.success { background: #e9f8f0; border-left: 3px solid #068906; }
    .status.error { background: #ffebee; border-left: 3px solid #f44336; }
    table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.85rem; }
    table tr {display: grid; grid-template-columns: 1fr 1fr 1fr; }
    th, td { text-align: left; padding: 0.5rem; border-bottom: 1px solid #eee; }
    th { color: #666; font-weight: 600; }
    pre { margin: 0.5rem 0; background: rgba(0, 47, 167, 0.02); padding: 0.5rem 1em; overflow-x: auto; font-size: 0.8rem; max-height: 200px; }
    #results { font-size: 0.85rem; margin-top: 1rem; }
  </style>
</head>
<body>
  <div class="container">
    <h1>${escapeHtml(title)}</h1>
    <p class="subtitle">Source format: ${graph.format} &middot; ${graph.nodes.length} operations &middot; ${graph.constants.length} constants</p>

    <div class="card">
      <div class="controls">
        <label for="device">Device:</label>
        <select id="device">
          <option value="cpu">CPU</option>
          <option value="gpu">GPU</option>
          <option value="npu">NPU</option>
        </select>
        <button id="runBtn" onclick="run()">Run Inference</button>
      </div>
      <div id="status" class="status info">Ready. Select a device and click Run.</div>
    </div>

    <div class="card">
      <h3>Model Info</h3>
      <div class="model-info">
        <table>
          <tr><th>Inputs</th><th>Shape</th><th>Data Type</th></tr>
  ${graph.inputs.map((i) => `        <tr><td>${escapeHtml(i.name)}</td><td>${JSON.stringify(i.shape)}</td><td>${i.dataType}</td></tr>`).join('\n')}
        </table>
        <table>
          <tr><th>Outputs</th><th>Shape</th><th>Data Type</th></tr>
  ${effectiveOutputs.map((o) => `        <tr><td>${escapeHtml(o.name)}</td><td>${JSON.stringify(o.shape)}</td><td>${o.dataType}</td></tr>`).join('\n')}
        </table>
      </div>
    </div>

    <div class="card" id="resultsCard" style="display:none;">
      <h3>Results</h3>
      <div id="results"></div>
    </div>
  </div>

  <script>
// ---- Generated WebNN Code ----
${jsCode}
// ---- Harness ----
const INPUT_INFO = [
${inputInfoLines.join('\n')}
];
const OUTPUT_INFO = [
${outputInfoLines.join('\n')}
];

async function run() {
  const statusEl = document.getElementById('status');
  const runBtn = document.getElementById('runBtn');
  const resultsCard = document.getElementById('resultsCard');
  const resultsEl = document.getElementById('results');
  const deviceType = document.getElementById('device').value;

  try {
    runBtn.disabled = true;
    statusEl.className = 'status info';
    statusEl.textContent = 'Checking WebNN support...';

    if (!navigator.ml) {
      throw new Error('WebNN is not supported in this browser. Try Chrome 147+ with WebNN flags enabled.');
    }

    statusEl.textContent = 'Creating context (' + deviceType + ')...';
    const context = await navigator.ml.createContext({ deviceType });

    statusEl.textContent = 'Loading weights...';
    const weights = await WeightsFile.load('${weightsFileName}', '${manifestFileName}');

    statusEl.textContent = 'Building graph...';
    const buildStart = performance.now();
    const graph = await buildGraph(context, weights);
    const buildElapsed = (performance.now() - buildStart).toFixed(2);
    console.log('Graph build: ' + buildElapsed + 'ms on ' + deviceType.toUpperCase());

    // Create input tensors
    const inputs = {};
    for (const info of INPUT_INFO) {
      const data = new info.TypedArray(info.size);
      const isBigInt = info.dataType === 'int64' || info.dataType === 'uint64';
      for (let i = 0; i < data.length; i++) data[i] = isBigInt ? BigInt(Math.floor(Math.random() * 100)) : Math.random() * 2 - 1;
      const tensor = await context.createTensor({
        dataType: info.dataType, shape: info.shape.map(d => typeof d === 'number' ? d : 1),
        writable: true
      });
      context.writeTensor(tensor, data);
      inputs[info.name] = tensor;
    }

    // Create output tensors — use actual shapes from built graph
    const outputs = {};
    for (const info of OUTPUT_INFO) {
      outputs[info.name] = await context.createTensor({
        dataType: info.dataType, shape: graph.outputShapes[info.name],
        readable: true
      });
    }

    statusEl.textContent = 'Running inference...';
    const t0 = performance.now();
    context.dispatch(graph.graph, inputs, outputs);

    // Read results
    const results = {};
    for (const info of OUTPUT_INFO) {
      const buf = await context.readTensor(outputs[info.name]);
      results[info.name] = new info.TypedArray(buf);
    }
    const elapsed = (performance.now() - t0).toFixed(2);

    // Benchmark: 50 runs
    const NUM_RUNS = 50;
    const runTimes = [];
    for (let i = 0; i < NUM_RUNS; i++) {
      const t0b = performance.now();
      context.dispatch(graph.graph, inputs, outputs);
      for (const info of OUTPUT_INFO) {
        await context.readTensor(outputs[info.name]);
      }
      runTimes.push(performance.now() - t0b);
    }
    const avgTime = (runTimes.reduce((a, b) => a + b, 0) / NUM_RUNS).toFixed(2);
    const sorted = [...runTimes].sort((a, b) => a - b);
    const medianTime = (NUM_RUNS % 2 ? sorted[NUM_RUNS >> 1] : (sorted[NUM_RUNS / 2 - 1] + sorted[NUM_RUNS / 2]) / 2).toFixed(2);
    console.log('Inference: ' + avgTime + 'ms (average \u00b7 ' + NUM_RUNS + ' runs) on ' + deviceType.toUpperCase());
    console.log('Inference: ' + medianTime + 'ms (median \u00b7 ' + NUM_RUNS + ' runs) on ' + deviceType.toUpperCase());

    statusEl.className = 'status success';
    statusEl.style.whiteSpace = 'pre-line';
    statusEl.textContent = 'Graph build: ' + buildElapsed + 'ms on ' + deviceType.toUpperCase()
      + '\\nInference: ' + elapsed + 'ms (1 run) on ' + deviceType.toUpperCase()
      + '\\nInference: ' + avgTime + 'ms (average \\u00b7 ' + NUM_RUNS + ' runs) on ' + deviceType.toUpperCase()
      + '\\nInference: ' + medianTime + 'ms (median \\u00b7 ' + NUM_RUNS + ' runs) on ' + deviceType.toUpperCase();

    // Show results
    resultsCard.style.display = '';
    let html = '';
    for (const info of OUTPUT_INFO) {
      const data = results[info.name];
      const preview = Array.from(data.slice(0, 10)).map(v => typeof v === 'bigint' ? v.toString() : Number(v).toFixed(6)).join(', ');
      html += '<div>' + info.name + ' ' + JSON.stringify(info.shape) + ' (' + info.dataType + ')</div>';
      html += '<pre>[' + preview + (data.length > 10 ? ', ...' : '') + ']</pre>';
    }
    resultsEl.innerHTML = html;

  } catch (err) {
    statusEl.className = 'status error';
    statusEl.textContent = 'Error: ' + err.message;
    console.error(err);
  } finally {
    runBtn.disabled = false;
  }
}
  </script>
</body>
</html>
`;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function escapeSingleQuotes(s: string): string {
  return s.replace(/'/g, "\\'");
}
