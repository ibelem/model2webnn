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
  statusEl.textContent = 'Fetching model from URL...';

  try {
    const fetchUrl = transformHuggingFaceUrl(url);
    const response = await fetch(fetchUrl);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const buffer = new Uint8Array(arrayBuffer);
    const urlPath = new URL(url).pathname;
    const fileName = urlPath.split('/').pop() ?? 'model.onnx';

    statusEl.textContent = `Downloaded: ${fileName} (${(buffer.byteLength / 1024 / 1024).toFixed(2)} MB)`;

    // Check for external data refs
    if (fileName.endsWith('.onnx')) {
      const refs = getExternalDataRefs(buffer);
      if (refs.length > 0) {
        // Try to fetch external data files from same URL base
        const baseUrl = fetchUrl.substring(0, fetchUrl.lastIndexOf('/') + 1);
        statusEl.textContent = `Fetching ${refs.length} external data file(s)...`;

        const externalData: ExternalDataMap = new Map();
        let allFetched = true;

        for (const ref of refs) {
          try {
            const refUrl = baseUrl + ref;
            const refResponse = await fetch(transformHuggingFaceUrl(refUrl));
            if (!refResponse.ok) {
              allFetched = false;
              break;
            }
            const refBuffer = new Uint8Array(await refResponse.arrayBuffer());
            externalData.set(ref, refBuffer);
            statusEl.textContent = `Fetched external data: ${ref} (${(refBuffer.byteLength / 1024 / 1024).toFixed(2)} MB)`;
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
  statusEl.textContent = `Loading: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)...`;

  const buffer = new Uint8Array(await file.arrayBuffer());

  if (file.name.endsWith('.onnx')) {
    const refs = getExternalDataRefs(buffer);
    if (refs.length > 0) {
      // Model needs external data — store pending and prompt user
      pendingModel = { buffer, fileName: file.name };
      pendingRefs = refs;
      statusEl.textContent = `Model loaded. External data files required (${refs.length} file(s)).`;
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

  statusEl.textContent = `Loading: ${modelFile.name} (${(modelFile.size / 1024 / 1024).toFixed(2)} MB)...`;
  const modelBuffer = new Uint8Array(await modelFile.arrayBuffer());

  // Check for external data refs
  const dataFiles = files.filter((f) => f !== modelFile);
  if (modelFile.name.endsWith('.onnx') && dataFiles.length > 0) {
    const refs = getExternalDataRefs(modelBuffer);
    if (refs.length > 0) {
      statusEl.textContent = `Loading ${dataFiles.length} external data file(s)...`;

      const externalData: ExternalDataMap = new Map();
      for (const f of dataFiles) {
        // Use the file name (strip path for folder uploads)
        const name = f.name;
        const data = new Uint8Array(await f.arrayBuffer());
        externalData.set(name, data);
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
        statusEl.textContent = `Model loaded. Still need ${missing.length} external data file(s).`;
        onExternalDataNeeded?.(missing);
        return;
      }

      const totalSize = [...externalData.values()].reduce((s, d) => s + d.byteLength, 0);
      statusEl.textContent = `Loaded: ${modelFile.name} + ${dataFiles.length} external data file(s) (${(totalSize / 1024 / 1024).toFixed(2)} MB total data)`;
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
  statusEl.textContent = `Loading ${files.length} external data file(s)...`;

  for (const f of files) {
    const data = new Uint8Array(await f.arrayBuffer());
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
    if (parsed.hostname === 'huggingface.co') {
      return url.replace('/blob/', '/resolve/');
    }
  } catch {
    // Not a valid URL, return as-is
  }
  return url;
}
