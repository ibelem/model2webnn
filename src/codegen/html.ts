// HTML code generator — emits a self-contained runnable .html file
// Embeds the generated JS code + WebNN test harness with device selector.

import type { GraphIR } from '../ir/graph.js';
import { getTypedArrayName } from '../ir/graph.js';
import { generateJavaScriptFixed } from './javascript.js';

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
    title = `WebNN — ${graph.name}`,
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

  const outputInfoLines: string[] = [];
  for (const output of graph.outputs) {
    const typedArray = getTypedArrayName(output.dataType);
    const numericShape = output.shape.map((d) => (typeof d === 'number' ? d : 1));
    const totalSize = numericShape.reduce((a, b) => a * b, 1);
    outputInfoLines.push(
      `    { name: '${escapeSingleQuotes(output.name)}', dataType: '${output.dataType}', shape: ${JSON.stringify(output.shape)}, TypedArray: ${typedArray}, size: ${totalSize} },`,
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
    body { font-family: system-ui, -apple-system, sans-serif; background: #f8f9fa; color: #1a1a1a; padding: 2rem; }
    .container { max-width: 900px; margin: 0 auto; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    .subtitle { color: #666; margin-bottom: 1.5rem; font-size: 0.9rem; }
    .card { background: #fff; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .controls { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
    label { font-weight: 600; font-size: 0.9rem; }
    select { padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; font-size: 0.9rem; }
    button { background: #0066cc; color: #fff; border: none; padding: 0.6rem 1.5rem; border-radius: 4px; font-size: 0.9rem; cursor: pointer; }
    button:hover { background: #0052a3; }
    button:disabled { background: #ccc; cursor: not-allowed; }
    .status { padding: 1rem; border-radius: 4px; margin-top: 1rem; font-size: 0.9rem; }
    .status.info { background: #e3f2fd; border-left: 3px solid #2196F3; }
    .status.success { background: #e8f5e9; border-left: 3px solid #4CAF50; }
    .status.error { background: #ffebee; border-left: 3px solid #f44336; }
    table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.85rem; }
    th, td { text-align: left; padding: 0.5rem; border-bottom: 1px solid #eee; }
    th { color: #666; font-weight: 600; }
    pre { background: #f5f5f5; padding: 1rem; border-radius: 4px; overflow-x: auto; font-size: 0.8rem; max-height: 200px; }
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
      <h3 style="margin-bottom: 0.5rem;">Model Info</h3>
      <table>
        <tr><th>Inputs</th><th>Shape</th><th>Data Type</th></tr>
${graph.inputs.map((i) => `        <tr><td>${escapeHtml(i.name)}</td><td>${JSON.stringify(i.shape)}</td><td>${i.dataType}</td></tr>`).join('\n')}
      </table>
      <table style="margin-top: 1rem;">
        <tr><th>Outputs</th><th>Shape</th><th>Data Type</th></tr>
${graph.outputs.map((o) => `        <tr><td>${escapeHtml(o.name)}</td><td>${JSON.stringify(o.shape)}</td><td>${o.dataType}</td></tr>`).join('\n')}
      </table>
    </div>

    <div class="card" id="resultsCard" style="display:none;">
      <h3 style="margin-bottom: 0.5rem;">Results</h3>
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
      throw new Error('WebNN is not supported in this browser. Try Chrome 131+ with WebNN flags enabled.');
    }

    statusEl.textContent = 'Creating context (' + deviceType + ')...';
    const context = await navigator.ml.createContext({ deviceType });

    statusEl.textContent = 'Loading weights...';
    const weights = await WeightsFile.load('${weightsFileName}', '${manifestFileName}');

    statusEl.textContent = 'Building graph...';
    const graph = await buildGraph(context, weights);

    // Create random inputs
    const inputs = {};
    for (const info of INPUT_INFO) {
      const arr = new info.TypedArray(info.size);
      for (let i = 0; i < arr.length; i++) arr[i] = Math.random() * 2 - 1;
      inputs[info.name] = arr;
    }

    // Create output buffers
    const outputs = {};
    for (const info of OUTPUT_INFO) {
      outputs[info.name] = new info.TypedArray(info.size);
    }

    statusEl.textContent = 'Running inference...';
    const t0 = performance.now();
    const results = await context.compute(graph, inputs, outputs);
    const elapsed = (performance.now() - t0).toFixed(2);

    statusEl.className = 'status success';
    statusEl.textContent = 'Inference completed in ' + elapsed + 'ms on ' + deviceType.toUpperCase();

    // Show results
    resultsCard.style.display = '';
    let html = '';
    for (const info of OUTPUT_INFO) {
      const data = results.outputs[info.name];
      const preview = Array.from(data.slice(0, 10)).map(v => typeof v === 'bigint' ? v.toString() : Number(v).toFixed(6)).join(', ');
      html += '<h4>' + info.name + ' ' + JSON.stringify(info.shape) + ' (' + info.dataType + ')</h4>';
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
