// Web UI — main application entry point
// Orchestrates upload, conversion, preview, and download.

import { convert, detectFormat } from '../index.js';
import { parseOnnx, type ExternalDataMap } from '../parsers/onnx.js';
import { parseTflite } from '../parsers/tflite.js';
import { getFreeDimensions } from '../ir/graph.js';
import { initUpload, showExternalDataPrompt, fetchFromUrl } from './upload.js';
import { initPreview } from './preview.js';
import { initDownload } from './download.js';

/** Pending state when free dimension overrides are needed */
let pendingConversion: {
  buffer: Uint8Array;
  fileName: string;
  externalData?: ExternalDataMap;
  freeDims: string[];
} | null = null;

/** True when the current model was fetched from a URL (enables dim↔URL sync). */
let modelFromUrl = false;

async function handleModelLoaded(
  buffer: Uint8Array,
  fileName: string,
  externalData?: ExternalDataMap,
): Promise<void> {
  const statusEl = document.getElementById('uploadStatus')!;

  try {
    statusEl.style.display = '';
    statusEl.className = 'status-banner info';
    statusEl.textContent = 'Analyzing model...';

    // Reset state from any previous model
    pendingConversion = null;
    // clearUrlParam() runs before onModelLoaded for local files, so 'url' is gone by now.
    modelFromUrl = new URLSearchParams(window.location.search).has('url');
    document.getElementById('freeDimSection')!.style.display = 'none';
    document.getElementById('externalDataSection')!.style.display = 'none';
    document.getElementById('coverageWarning')!.style.display = 'none';
    document.getElementById('previewSection')!.style.display = 'none';
    document.getElementById('downloadSection')!.style.display = 'none';

    // Transition from centered upload to sidebar layout
    document.querySelector('main')!.classList.add('has-model');

    // Quick-parse to detect free dimensions
    const format = detectFormat(buffer);
    let freeDims: string[] = [];

    if (format === 'onnx') {
      const graph = await parseOnnx(buffer, externalData);
      freeDims = getFreeDimensions(graph);
    } else if (format === 'tflite') {
      const graph = await parseTflite(buffer);
      freeDims = getFreeDimensions(graph);
    }

    let urlDimOverrides: Record<string, number> | undefined;
    if (freeDims.length > 0) {
      // Has dynamic dimensions — show override UI
      pendingConversion = { buffer, fileName, externalData, freeDims };
      showFreeDimOverrideUI(freeDims);
      // Pre-populate inputs from URL dim params (e.g. &dim=batch_size:1&dim=seq_len:128)
      if (modelFromUrl) {
        urlDimOverrides = populateDimsFromUrl(freeDims);
      }
    }

    // Convert immediately (with URL dim overrides if present, otherwise symbolic dims)
    await runConversion(buffer, fileName, externalData, urlDimOverrides);
  } catch (err: unknown) {
    statusEl.className = 'status-banner error';
    statusEl.textContent = `Error: ${(err as Error).message}`;
  }
}

/** Debounce timer for free dimension input changes */
let freeDimDebounceTimer: ReturnType<typeof setTimeout> | null = null;

/**
 * Show the free dimension override UI controls.
 */
function updateFreeDimSectionStyle(): void {
  if (!pendingConversion) return;
  const section = document.getElementById('freeDimSection')!;
  const allSet = pendingConversion.freeDims.every((dim) => {
    const input = document.getElementById(`freeDim_${dim}`) as HTMLInputElement | null;
    const v = input?.value.trim();
    return v !== undefined && v !== '' && !isNaN(parseInt(v, 10)) && parseInt(v, 10) >= 1;
  });
  section.classList.toggle('sub-card--error', !allSet);
  section.classList.toggle('sub-card--info', allSet);
}

function showFreeDimOverrideUI(freeDims: string[]): void {
  const section = document.getElementById('freeDimSection')!;
  const list = document.getElementById('freeDimList')!;

  list.innerHTML = '';
  for (const dim of freeDims) {
    const row = document.createElement('div');
    row.className = 'free-dim-row';
    row.innerHTML = `
      <label for="freeDim_${dim}" title="${dim}">${dim}</label>
      <input type="number" id="freeDim_${dim}" name="freeDim_${dim}" min="1" placeholder="">
    `;
    list.appendChild(row);
  }

  section.style.display = '';
  updateFreeDimSectionStyle();

  // Auto-regenerate on input change (debounced)
  list.addEventListener('input', () => {
    updateFreeDimSectionStyle();
    syncDimsToUrl(); // update URL immediately for shareability
    if (freeDimDebounceTimer) clearTimeout(freeDimDebounceTimer);
    freeDimDebounceTimer = setTimeout(() => handleFreeDimConvert(), 600);
  });
}

/**
 * Handle the "Generate Code" button click after user sets free dimension values.
 */
async function handleFreeDimConvert(): Promise<void> {
  if (!pendingConversion) return;

  const statusEl = document.getElementById('uploadStatus')!;
  const overrides: Record<string, number> = {};

  for (const dim of pendingConversion.freeDims) {
    const input = document.getElementById(`freeDim_${dim}`) as HTMLInputElement | null;
    const value = input?.value.trim();

    if (!value || value === '') {
      // Not set — keep symbolic
      continue;
    }

    const num = parseInt(value, 10);
    if (isNaN(num) || num < 1) {
      statusEl.style.display = '';
      statusEl.className = 'status-banner error';
      statusEl.textContent = `Value for "${dim}" must be a positive integer.`;
      input?.focus();
      return;
    }

    overrides[dim] = num;
  }

  const { buffer, fileName, externalData } = pendingConversion;

  await runConversion(buffer, fileName, externalData, Object.keys(overrides).length > 0 ? overrides : undefined);
}

/**
 * Run the actual conversion and display results.
 */
async function runConversion(
  buffer: Uint8Array,
  fileName: string,
  externalData?: ExternalDataMap,
  freeDimensionOverrides?: Record<string, number>,
): Promise<void> {
  const statusEl = document.getElementById('uploadStatus')!;
  const previewSection = document.getElementById('previewSection')!;
  const downloadSection = document.getElementById('downloadSection')!;

  try {
    statusEl.style.display = '';
    statusEl.className = 'status-banner info';
    statusEl.textContent = 'Converting model...';

    const modelName = fileName.replace(/\.(onnx|tflite)$/i, '');

    const result = await convert(buffer, {
      format: 'javascript',
      weightsFileName: `${modelName}.weights`,
      manifestFileName: `${modelName}.manifest.json`,
      modelName,
      externalData,
      freeDimensionOverrides,
    });

    statusEl.className = 'status-banner success';
    const g = result.graph;
    const cov = result.coverage;
    let statusText = `Converted: ${g.name} — ${g.nodes.length} ops, ${g.constants.length} constants, ${(result.weights.byteLength / 1024).toFixed(0)} KB weights`;
    if (freeDimensionOverrides && Object.keys(freeDimensionOverrides).length > 0) {
      const dimStr = Object.entries(freeDimensionOverrides).map(([k, v]) => `${k}=${v}`).join(', ');
      statusText += ` (dimensions: ${dimStr})`;
    }
    if (cov.unsupportedOps > 0) {
      statusEl.className = 'status-banner info';
      statusText += ` — ${cov.coveragePercent}% coverage (${cov.unsupportedOps} unsupported ops)`;
    }
    statusEl.textContent = statusText;

    // Show unsupported ops warning if any
    if (cov.unsupportedOps > 0) {
      const warningEl = document.getElementById('coverageWarning')!;
      warningEl.style.display = '';
      const listEl = document.getElementById('unsupportedOpsList')!;
      listEl.innerHTML = cov.unsupportedOpTypes
        .map(({ opType, count }) => `<li>${opType} <span class="op-count">×${count}</span></li>`)
        .join('');
    } else {
      const warningEl = document.getElementById('coverageWarning')!;
      warningEl.style.display = 'none';
    }

    // Show preview
    previewSection.style.display = '';
    const graphInfoEl = document.getElementById('graphInfo')!;
    graphInfoEl.textContent = `${g.inputs.length} inputs · ${g.outputs.length} outputs · ${g.nodes.length} ops`;

    initPreview(result);

    // Show download
    downloadSection.style.display = '';
    initDownload(result, modelName);
  } catch (err: unknown) {
    statusEl.className = 'status-banner error';
    statusEl.textContent = `Error: ${(err as Error).message}`;
    previewSection.style.display = 'none';
    downloadSection.style.display = 'none';
  }
}

function handleExternalDataNeeded(refs: string[]): void {
  showExternalDataPrompt(refs);
}

/**
 * Read &dim=NAME:VALUE params from the current URL and populate the free dim inputs.
 * Splits on the FIRST colon only, so names like "serving_default_input:0" are safe.
 * Returns the parsed overrides (only dims that are valid positive integers).
 */
function populateDimsFromUrl(freeDims: string[]): Record<string, number> | undefined {
  const params = new URLSearchParams(window.location.search);
  const overrides: Record<string, number> = {};
  for (const raw of params.getAll('dim')) {
    const sep = raw.indexOf(':');
    if (sep < 1) continue;
    const name = raw.slice(0, sep);
    const val = parseInt(raw.slice(sep + 1), 10);
    if (!isNaN(val) && val >= 1 && freeDims.includes(name)) {
      const input = document.getElementById(`freeDim_${name}`) as HTMLInputElement | null;
      if (input) input.value = String(val);
      overrides[name] = val;
    }
  }
  updateFreeDimSectionStyle();
  return Object.keys(overrides).length > 0 ? overrides : undefined;
}

/**
 * Sync the current free dim input values back to the URL as &dim=NAME:VALUE params.
 * Only runs when the model was loaded from a URL (no-op for local files).
 */
function syncDimsToUrl(): void {
  if (!modelFromUrl || !pendingConversion) return;
  const url = new URL(location.href);
  url.searchParams.delete('dim');
  for (const dim of pendingConversion.freeDims) {
    const input = document.getElementById(`freeDim_${dim}`) as HTMLInputElement | null;
    const v = input?.value.trim();
    const n = v ? parseInt(v, 10) : NaN;
    if (!isNaN(n) && n >= 1) {
      url.searchParams.append('dim', `${dim}:${n}`);
    }
  }
  history.replaceState(null, '', url.toString());
}

// Initialize the app
initUpload(handleModelLoaded, handleExternalDataNeeded);

// Auto-fetch if ?url= query parameter is present
const urlParam = new URLSearchParams(window.location.search).get('url');
if (urlParam) {
  const netronLink = document.getElementById('netronLink') as HTMLAnchorElement | null;
  if (netronLink) {
    netronLink.href = `https://ibelem.github.io/netron/?url=${encodeURIComponent(urlParam)}`;
  }
  fetchFromUrl(handleModelLoaded, handleExternalDataNeeded, urlParam);
}
