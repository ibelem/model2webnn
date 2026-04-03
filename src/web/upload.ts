// Upload module — handles file drag-drop, file picker, URL fetch,
// and external data file loading for ONNX models.

import { getExternalDataRefs, type ExternalDataMap } from '../parsers/onnx.js';

type OnModelLoaded = (
  buffer: Uint8Array,
  fileName: string,
  externalData?: ExternalDataMap,
) => Promise<void>;

/** Callback for when external data refs are detected but not yet provided */
type OnExternalDataNeeded = (refs: string[]) => void;

/** State held between model load and external data upload */
let pendingModel: { buffer: Uint8Array; fileName: string } | null = null;
let pendingRefs: string[] = [];
let currentCallback: OnModelLoaded | null = null;

// --------------- Progress bar helpers ---------------

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/** Create (or reuse) a progress bar inside a status element. Returns update/complete callbacks. */
function createProgressBar(
  container: HTMLElement,
  fileName: string,
): {
  update: (loaded: number, total: number | null) => void;
  complete: (finalSize: number) => void;
  el: HTMLElement;
} {
  container.style.display = '';
  container.className = 'status-banner info';

  const wrapper = document.createElement('div');
  wrapper.className = 'progress-container';
  wrapper.innerHTML = `
    <div class="progress-label">
      <span class="progress-filename">${escapeHtml(fileName)}</span>
      <span class="progress-stats">0%</span>
    </div>
    <div class="progress-track">
      <div class="progress-fill" style="width: 0%"></div>
    </div>`;

  container.innerHTML = '';
  container.appendChild(wrapper);

  const statsEl = wrapper.querySelector('.progress-stats')!;
  const fillEl = wrapper.querySelector('.progress-fill') as HTMLElement;

  // If total is unknown, start as indeterminate
  let wasIndeterminate = false;

  return {
    el: wrapper,
    update(loaded: number, total: number | null) {
      if (total && total > 0) {
        if (wasIndeterminate) {
          fillEl.classList.remove('indeterminate');
          wasIndeterminate = false;
        }
        const pct = Math.min(100, (loaded / total) * 100);
        fillEl.style.width = `${pct.toFixed(1)}%`;
        statsEl.textContent = `${formatBytes(loaded)} / ${formatBytes(total)}  ·  ${pct.toFixed(0)}%`;
      } else {
        if (!wasIndeterminate) {
          fillEl.classList.add('indeterminate');
          wasIndeterminate = true;
        }
        statsEl.textContent = formatBytes(loaded);
      }
    },
    complete(finalSize: number) {
      fillEl.classList.remove('indeterminate');
      fillEl.classList.add('done');
      fillEl.style.width = '100%';
      statsEl.textContent = formatBytes(finalSize);
    },
  };
}

function escapeHtml(s: string): string {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/**
 * Fetch a URL with streaming progress. Returns the final Uint8Array.
 * Pre-allocates when Content-Length is available to avoid double memory usage.
 */
async function fetchWithProgress(
  url: string,
  progress: { update: (loaded: number, total: number | null) => void },
): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const contentLength = response.headers.get('Content-Length');
  const total = contentLength ? parseInt(contentLength, 10) : null;

  if (!response.body) {
    // Fallback: no streaming support
    const buf = new Uint8Array(await response.arrayBuffer());
    progress.update(buf.byteLength, buf.byteLength);
    return buf;
  }

  const reader = response.body.getReader();

  if (total && total > 0) {
    // Known size: pre-allocate and write directly — avoids double memory usage
    const result = new Uint8Array(total);
    let offset = 0;

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      result.set(value, offset);
      offset += value.byteLength;
      progress.update(offset, total);
    }

    return offset === total ? result : result.slice(0, offset);
  }

  // Unknown size: accumulate chunks then merge
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    progress.update(loaded, null);
  }

  const result = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

/**
 * Read a File with progress tracking.
 */
function readFileWithProgress(
  file: File,
  progress: { update: (loaded: number, total: number | null) => void },
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onprogress = (e) => {
      progress.update(e.loaded, e.lengthComputable ? e.total : file.size);
    };
    reader.onload = () => {
      const buf = new Uint8Array(reader.result as ArrayBuffer);
      progress.update(buf.byteLength, buf.byteLength);
      resolve(buf);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}

export function initUpload(
  onModelLoaded: OnModelLoaded,
  onExternalDataNeeded?: OnExternalDataNeeded,
): void {
  const dropZone = document.getElementById('dropZone')!;
  const fileInput = document.getElementById('fileInput') as HTMLInputElement;
  const folderInput = document.getElementById('folderInput') as HTMLInputElement;
  const fetchUrlBtn = document.getElementById('fetchUrlBtn')!;
  const externalDataInput = document.getElementById('externalDataInput') as HTMLInputElement;

  currentCallback = onModelLoaded;

  // Drag and drop — supports multiple files
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      await handleMultipleFiles(Array.from(files), onModelLoaded, onExternalDataNeeded);
    }
  });

  // File picker (single model file)
  fileInput.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    if (file) {
      await handleSingleModelFile(file, onModelLoaded, onExternalDataNeeded);
      fileInput.value = '';
    }
  });

  // Folder picker — user selects folder containing model + external data
  folderInput.addEventListener('change', async () => {
    const files = folderInput.files;
    if (files && files.length > 0) {
      await handleMultipleFiles(Array.from(files), onModelLoaded, onExternalDataNeeded);
      folderInput.value = '';
    }
  });

  // External data file picker — follow-up upload for missing external data
  externalDataInput.addEventListener('change', async () => {
    const files = externalDataInput.files;
    if (files && files.length > 0) {
      await handleExternalDataFiles(Array.from(files));
      externalDataInput.value = '';
    }
  });

  // URL fetch
  fetchUrlBtn.addEventListener('click', () => fetchFromUrl(onModelLoaded, onExternalDataNeeded));
}

/**
 * Fetch a model from the URL in the input field (or a provided URL).
 * Exported so it can be triggered programmatically (e.g. from ?url= query param).
 */
export async function fetchFromUrl(
  onModelLoaded: OnModelLoaded,
  onExternalDataNeeded?: OnExternalDataNeeded,
  urlOverride?: string,
): Promise<void> {
  const modelUrlInput = document.getElementById('modelUrl') as HTMLInputElement;
  const statusEl = document.getElementById('uploadStatus')!;

  const url = urlOverride ?? modelUrlInput.value.trim();
  if (!url) return;

  // Fill the input so the user sees the URL
  if (urlOverride) modelUrlInput.value = urlOverride;

  statusEl.style.display = '';
  statusEl.className = 'status-banner info';
  statusEl.innerHTML = '<svg class="spinner" width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6.5" stroke="currentColor" stroke-width="1.5" stroke-dasharray="32" stroke-dashoffset="8" stroke-linecap="round"/></svg> Fetching model from URL...';

  try {
    const fetchUrl = transformHuggingFaceUrl(url);
    let effectiveUrl = fetchUrl;
    let buffer: Uint8Array;

    const urlPath = new URL(url).pathname;
    const fileName = urlPath.split('/').pop() ?? 'model.onnx';
    const pb = createProgressBar(statusEl, fileName);

    // For HuggingFace URLs, do a fast HEAD pre-check to avoid a slow
    // GET timeout before falling back to hf-mirror.com.
    const mirrorUrl = getHuggingFaceMirrorUrl(url);
    let useMirror = false;
    if (mirrorUrl) {
      const reachable = await isReachableViaHead(fetchUrl);
      if (!reachable) {
        useMirror = true;
      }
    }

    if (useMirror) {
      const mirrorPb = createProgressBar(statusEl, `${fileName} (via hf-mirror.com)`);
      effectiveUrl = transformHuggingFaceUrl(mirrorUrl!);
      buffer = await fetchWithProgress(effectiveUrl, mirrorPb);
      mirrorPb.complete(buffer.byteLength);
    } else {
      try {
        buffer = await fetchWithProgress(fetchUrl, pb);
      } catch (primaryErr) {
        // GET failed — try mirror as last resort
        if (mirrorUrl) {
          const mirrorPb = createProgressBar(statusEl, `${fileName} (via hf-mirror.com)`);
          effectiveUrl = transformHuggingFaceUrl(mirrorUrl);
          buffer = await fetchWithProgress(effectiveUrl, mirrorPb);
          mirrorPb.complete(buffer.byteLength);
        } else {
          throw primaryErr;
        }
      }
    }

    pb.complete(buffer.byteLength);
    const host = new URL(effectiveUrl).hostname;
    statusEl.querySelector('.progress-filename')!.textContent = `${fileName} — ${formatBytes(buffer.byteLength)} from ${host}`;

    // Detect Git LFS pointer files returned by raw.githubusercontent.com
    const LFS_MAGIC = 'version https://git-lfs.github.com/spec/v1';
    const prefix = new TextDecoder().decode(buffer.slice(0, LFS_MAGIC.length));
    if (prefix === LFS_MAGIC) {
      throw new Error(
        'This file is stored in Git LFS — GitHub returns a pointer file instead of the real content. ' +
        'Use a HuggingFace URL, download via "git lfs pull", or find a direct CDN link.',
      );
    }

    // Check for external data refs
    if (fileName.endsWith('.onnx')) {
      const refs = getExternalDataRefs(buffer);
      if (refs.length > 0) {
        // Try to fetch external data files from same URL base
        const baseUrl = effectiveUrl.substring(0, effectiveUrl.lastIndexOf('/') + 1);

        const externalData: ExternalDataMap = new Map();
        let allFetched = true;

        for (const ref of refs) {
          try {
            const refPb = createProgressBar(statusEl, ref);
            const refUrl = baseUrl + ref;
            const refBuffer = await fetchWithProgress(transformHuggingFaceUrl(refUrl), refPb);
            refPb.complete(refBuffer.byteLength);
            externalData.set(ref, refBuffer);
          } catch {
            allFetched = false;
            break;
          }
        }

        if (allFetched) {
          await onModelLoaded(buffer, fileName, externalData);
        } else {
          // Could not auto-fetch; prompt user to upload manually
          pendingModel = { buffer, fileName };
          pendingRefs = refs;
          onExternalDataNeeded?.(refs);
        }
        return;
      }
    }

    await onModelLoaded(buffer, fileName);
  } catch (err: unknown) {
    statusEl.className = 'status-banner error';
    statusEl.textContent = `Fetch error: ${(err as Error).message}`;
  }
}

/**
 * Handle a single model file upload. Checks for external data refs and either
 * proceeds directly or prompts for additional files.
 */
async function handleSingleModelFile(
  file: File,
  onModelLoaded: OnModelLoaded,
  onExternalDataNeeded?: OnExternalDataNeeded,
): Promise<void> {
  const statusEl = document.getElementById('uploadStatus')!;
  statusEl.style.display = '';
  statusEl.className = 'status-banner info';

  const pb = createProgressBar(statusEl, file.name);
  const buffer = await readFileWithProgress(file, pb);
  pb.complete(buffer.byteLength);

  if (file.name.endsWith('.onnx')) {
    const refs = getExternalDataRefs(buffer);
    if (refs.length > 0) {
      // Model needs external data — store pending and prompt user
      pendingModel = { buffer, fileName: file.name };
      pendingRefs = refs;
      statusEl.querySelector('.progress-filename')!.textContent = `${file.name} — external data files required (${refs.length} file(s))`;
      onExternalDataNeeded?.(refs);
      return;
    }
  }

  await onModelLoaded(buffer, file.name);
}

/**
 * Handle multiple files from drag-drop or folder upload.
 * Detects the .onnx model file and treats the rest as external data.
 */
async function handleMultipleFiles(
  files: File[],
  onModelLoaded: OnModelLoaded,
  onExternalDataNeeded?: OnExternalDataNeeded,
): Promise<void> {
  const statusEl = document.getElementById('uploadStatus')!;
  statusEl.style.display = '';
  statusEl.className = 'status-banner info';

  // Find the model file
  const modelFile = files.find((f) => {
    const name = f.name.toLowerCase();
    return name.endsWith('.onnx') || name.endsWith('.tflite');
  });

  if (!modelFile) {
    statusEl.className = 'status-banner error';
    statusEl.textContent = 'No .onnx or .tflite model file found in the selected files.';
    return;
  }

  const pb = createProgressBar(statusEl, modelFile.name);
  const modelBuffer = await readFileWithProgress(modelFile, pb);
  pb.complete(modelBuffer.byteLength);

  // Check for external data refs
  const dataFiles = files.filter((f) => f !== modelFile);
  if (modelFile.name.endsWith('.onnx') && dataFiles.length > 0) {
    const refs = getExternalDataRefs(modelBuffer);
    if (refs.length > 0) {
      const externalData: ExternalDataMap = new Map();
      for (const f of dataFiles) {
        const dataPb = createProgressBar(statusEl, f.name);
        const data = await readFileWithProgress(f, dataPb);
        dataPb.complete(data.byteLength);
        externalData.set(f.name, data);
      }

      // Check if all refs are satisfied
      const missing = refs.filter((r) => !externalData.has(r));
      if (missing.length > 0) {
        // Some files still missing — store pending and prompt
        pendingModel = { buffer: modelBuffer, fileName: modelFile.name };
        pendingRefs = missing;
        // Still store what we have so far
        for (const [k, v] of externalData) {
          pendingExternalData.set(k, v);
        }
        statusEl.querySelector('.progress-filename')!.textContent = `${modelFile.name} — still need ${missing.length} external data file(s)`;
        onExternalDataNeeded?.(missing);
        return;
      }

      const totalSize = [...externalData.values()].reduce((s, d) => s + d.byteLength, 0);
      statusEl.querySelector('.progress-filename')!.textContent = `${modelFile.name} + ${dataFiles.length} file(s) — ${formatBytes(totalSize)} total`;
      await onModelLoaded(modelBuffer, modelFile.name, externalData);
      return;
    }
  }

  // No external data needed or not an ONNX model
  await onModelLoaded(modelBuffer, modelFile.name);
}

/** Accumulated external data for multi-step uploads */
const pendingExternalData: ExternalDataMap = new Map();

/**
 * Handle external data files uploaded after the initial model load.
 */
async function handleExternalDataFiles(files: File[]): Promise<void> {
  const statusEl = document.getElementById('externalDataStatus')!;
  statusEl.style.display = '';
  statusEl.className = 'status-banner info';

  for (const f of files) {
    const pb = createProgressBar(statusEl, f.name);
    const data = await readFileWithProgress(f, pb);
    pb.complete(data.byteLength);
    pendingExternalData.set(f.name, data);
  }

  // Check if all pending refs are now satisfied
  const stillMissing = pendingRefs.filter((r) => !pendingExternalData.has(r));
  if (stillMissing.length > 0) {
    statusEl.textContent = `Still need ${stillMissing.length} file(s): ${stillMissing.join(', ')}`;
    // Update the list
    updateExternalDataList(stillMissing);
    return;
  }

  // All files loaded — proceed with conversion
  statusEl.className = 'status-banner success';
  statusEl.textContent = 'All external data files loaded.';

  if (pendingModel && currentCallback) {
    const externalData = new Map(pendingExternalData);
    const { buffer, fileName } = pendingModel;

    // Reset pending state
    pendingModel = null;
    pendingRefs = [];
    pendingExternalData.clear();

    // Hide external data section
    const section = document.getElementById('externalDataSection')!;
    section.style.display = 'none';

    await currentCallback(buffer, fileName, externalData);
  }
}

function updateExternalDataList(refs: string[]): void {
  const list = document.getElementById('externalDataList')!;
  list.innerHTML = '';
  for (const ref of refs) {
    const li = document.createElement('li');
    li.textContent = ref;
    list.appendChild(li);
  }
}

/**
 * Show the external data prompt section with the list of required files.
 * Called from app.ts when external data is needed.
 */
export function showExternalDataPrompt(refs: string[]): void {
  const section = document.getElementById('externalDataSection')!;
  section.style.display = '';
  updateExternalDataList(refs);
}

/**
 * Transform HuggingFace model page URLs into direct download URLs.
 */
function transformHuggingFaceUrl(url: string): string {
  try {
    const parsed = new URL(url);

    // GitHub raw file URLs redirect through github.com which strips CORS headers.
    // Rewrite directly to raw.githubusercontent.com to avoid the redirect.
    // github.com/:owner/:repo/raw/:branch/:path
    //   → raw.githubusercontent.com/:owner/:repo/:branch/:path
    if (parsed.hostname === 'github.com') {
      const m = parsed.pathname.match(/^\/([^/]+)\/([^/]+)\/raw\/(.+)$/);
      if (m) {
        return `https://raw.githubusercontent.com/${m[1]}/${m[2]}/${m[3]}`;
      }
    }

    if (parsed.hostname === 'huggingface.co' || parsed.hostname === 'hf-mirror.com') {
      return url.replace('/blob/', '/resolve/');
    }
  } catch {
    // Not a valid URL, return as-is
  }
  return url;
}

/**
 * Get the hf-mirror.com fallback URL for a HuggingFace URL.
 */
function getHuggingFaceMirrorUrl(url: string): string | null {
  try {
    const parsed = new URL(url);
    if (parsed.hostname === 'huggingface.co') {
      parsed.hostname = 'hf-mirror.com';
      return parsed.toString();
    }
  } catch {
    // ignore
  }
  return null;
}

/**
 * Quick reachability check using HTTP HEAD with a short timeout.
 * Returns true if the host responds (any 2xx/3xx), false on timeout or network error.
 */
async function isReachableViaHead(url: string, timeoutMs = 5000): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { method: 'HEAD', signal: controller.signal });
    return res.ok || (res.status >= 300 && res.status < 400);
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}
