// Runner module — live preview of generated HTML + JS + weights in a sandboxed iframe
// Uses blob URLs to inject the weights/manifest so fetch() calls resolve.
// Blob URLs are created lazily on first run to avoid memory overhead for large models.

import type { ConvertResult } from '../index.js';

const ALLOWED_DEVICES = new Set(['cpu', 'gpu', 'npu']);

let currentResult: ConvertResult | null = null;
let weightsUrl: string | null = null;
let manifestUrl: string | null = null;

export function initRunner(result: ConvertResult): void {
  currentResult = result;

  // Revoke previous blob URLs (free memory from prior model)
  revokeBlobUrls();

  // Wire up the Run button
  const runBtn = document.getElementById('previewRunBtn')!;
  const newBtn = runBtn.cloneNode(true) as HTMLElement;
  runBtn.parentNode!.replaceChild(newBtn, runBtn);
  newBtn.addEventListener('click', () => runPreview());

  // Set initial status
  const statusEl = document.getElementById('previewStatus')!;
  statusEl.textContent = 'Ready — click Run to execute';
}

function revokeBlobUrls(): void {
  if (weightsUrl) { URL.revokeObjectURL(weightsUrl); weightsUrl = null; }
  if (manifestUrl) { URL.revokeObjectURL(manifestUrl); manifestUrl = null; }
}

function ensureBlobUrls(): boolean {
  if (!currentResult) return false;
  if (weightsUrl && manifestUrl) return true;

  revokeBlobUrls();
  const weightsBlob = new Blob([currentResult.weights.buffer as ArrayBuffer], { type: 'application/octet-stream' });
  const manifestBlob = new Blob([JSON.stringify(currentResult.manifest)], { type: 'application/json' });
  weightsUrl = URL.createObjectURL(weightsBlob);
  manifestUrl = URL.createObjectURL(manifestBlob);
  return true;
}

function runPreview(): void {
  if (!currentResult || !ensureBlobUrls()) return;

  // Check for unresolved free dimensions (captured before they were defaulted to 1)
  const freeDims = currentResult.unresolvedFreeDims;
  if (freeDims.length > 0) {
    const dimList = freeDims.join(', ');
    const proceed = confirm(
      `This model has unresolved dynamic dimensions: ${dimList}\n\n` +
      `They will default to 1, which may cause errors.\n` +
      `Set values in the "Free dimension overrides" section for correct results.\n\n` +
      `Run anyway?`
    );
    if (!proceed) return;
  }

  const iframe = document.getElementById('previewFrame') as HTMLIFrameElement;
  const statusEl = document.getElementById('previewStatus')!;
  const deviceSelect = document.getElementById('previewDevice') as HTMLSelectElement;
  const deviceType = ALLOWED_DEVICES.has(deviceSelect.value) ? deviceSelect.value : 'cpu';

  statusEl.textContent = 'Loading...';

  // Rewrite the generated HTML to use blob URLs instead of relative file paths
  let html = currentResult.html ?? '';
  if (!html) {
    statusEl.textContent = 'Error: No HTML output available';
    return;
  }

  // ensureBlobUrls() guarantees these are non-null
  const wUrl = weightsUrl!;
  const mUrl = manifestUrl!;

  // Extract the model file names from the generated code
  const modelName = currentResult.graph.name;
  const weightsFileName = `${modelName}.weights`;
  const manifestFileName = `${modelName}.manifest.json`;

  // Replace fetch paths in the generated HTML/JS
  // The generated code uses: WeightsFile.load('model.weights', 'model.manifest.json')
  // We need to replace those with blob URLs
  html = html.replace(
    new RegExp(escapeRegExp(weightsFileName), 'g'),
    wUrl,
  );
  html = html.replace(
    new RegExp(escapeRegExp(manifestFileName), 'g'),
    mUrl,
  );

  // Also try generic names in case naming differs
  html = html.replace(
    /(['"])model\.weights\1/g,
    `'${wUrl}'`,
  );
  html = html.replace(
    /(['"])model\.manifest\.json\1/g,
    `'${mUrl}'`,
  );

  // Inject auto-run script: automatically click "Run Inference" with selected device
  // The generated HTML uses id="device" for the select and id="runBtn" for the button
  const safeDevice = deviceType.replace(/[^a-z]/g, '');
  const autoRunScript = `
<script>
  window.addEventListener('load', function() {
    setTimeout(function() {
      var sel = document.getElementById('device');
      if (sel) sel.value = '${safeDevice}';
      var btn = document.getElementById('runBtn');
      if (btn) btn.click();
    }, 300);
  });
</script>
</body>`;

  html = html.replace('</body>', autoRunScript);

  // Set iframe content
  iframe.srcdoc = html;

  // Listen for load
  iframe.onload = () => {
    statusEl.textContent = `Running on ${deviceType.toUpperCase()}...`;
    // After a timeout, update status (we can't easily get feedback from the iframe)
    setTimeout(() => {
      statusEl.textContent = `Loaded (${deviceType.toUpperCase()}) — check iframe for results`;
    }, 2000);
  };
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
