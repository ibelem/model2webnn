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
    document.getElementById('freeDimSection')!.style.display = 'none';
    document.getElementById('externalDataSection')!.style.display = 'none';
    document.getElementById('coverageWarning')!.style.display = 'none';
    document.getElementById('previewSection')!.style.display = 'none';
    document.getElementById('downloadSection')!.style.display = 'none';
    const emptyState = document.getElementById('emptyState');
    if (emptyState) emptyState.style.display = '';

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

    if (freeDims.length > 0) {
      // Has dynamic dimensions — show override UI, but convert immediately with symbolic dims
      pendingConversion = { buffer, fileName, externalData, freeDims };
      showFreeDimOverrideUI(freeDims);
    }

    // Convert immediately (symbolic dims kept as-is if no overrides)
    await runConversion(buffer, fileName, externalData);
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
function showFreeDimOverrideUI(freeDims: string[]): void {
  const section = document.getElementById('freeDimSection')!;
  const list = document.getElementById('freeDimList')!;

  list.innerHTML = '';
  for (const dim of freeDims) {
    const row = document.createElement('div');
    row.className = 'free-dim-row';
    row.innerHTML = `
      <label for="freeDim_${dim}">${dim}</label>
      <input type="number" id="freeDim_${dim}" name="freeDim_${dim}" min="1" placeholder="e.g. 1">
    `;
    list.appendChild(row);
  }

  section.style.display = '';

  // Auto-regenerate on input change (debounced)
  list.addEventListener('input', () => {
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
  const emptyState = document.getElementById('emptyState');

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
    if (emptyState) emptyState.style.display = 'none';
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
    if (emptyState) emptyState.style.display = '';
  }
}

function handleExternalDataNeeded(refs: string[]): void {
  showExternalDataPrompt(refs);
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
