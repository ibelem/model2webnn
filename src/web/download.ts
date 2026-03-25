// Download module — handles downloading generated files

import type { ConvertResult } from '../index.js';

let downloadResult: ConvertResult | null = null;
let downloadModelName = 'model';

export function initDownload(result: ConvertResult, modelName: string): void {
  downloadResult = result;
  downloadModelName = modelName;

  document.getElementById('downloadAllBtn')!.onclick = downloadAll;
  document.getElementById('downloadJsBtn')!.onclick = downloadJs;
  document.getElementById('downloadHtmlBtn')!.onclick = downloadHtml;
  document.getElementById('downloadWeightsBtn')!.onclick = downloadWeights;
}

function downloadBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function downloadJs(): void {
  if (!downloadResult) return;
  const blob = new Blob([downloadResult.code], { type: 'text/javascript' });
  downloadBlob(blob, `${downloadModelName}.js`);
}

function downloadHtml(): void {
  if (!downloadResult?.html) return;
  const blob = new Blob([downloadResult.html], { type: 'text/html' });
  downloadBlob(blob, `${downloadModelName}.html`);
}

function downloadWeights(): void {
  if (!downloadResult) return;
  const blob = new Blob([downloadResult.weights], { type: 'application/octet-stream' });
  downloadBlob(blob, `${downloadModelName}.weights`);
}

async function downloadAll(): Promise<void> {
  if (!downloadResult) return;

  // Use JSZip if available, otherwise download files individually
  try {
    const { default: JSZip } = await import('jszip');
    const zip = new JSZip();
    zip.file(`${downloadModelName}.js`, downloadResult.code);
    zip.file(`${downloadModelName}.weights`, downloadResult.weights);
    zip.file(`${downloadModelName}.manifest.json`, JSON.stringify(downloadResult.manifest, null, 2));
    if (downloadResult.html) {
      zip.file(`${downloadModelName}.html`, downloadResult.html);
    }

    const blob = await zip.generateAsync({ type: 'blob' });
    downloadBlob(blob, `${downloadModelName}-webnn.zip`);
  } catch {
    // Fallback: download files individually
    downloadJs();
    downloadHtml();
    downloadWeights();

    const manifestBlob = new Blob(
      [JSON.stringify(downloadResult.manifest, null, 2)],
      { type: 'application/json' },
    );
    downloadBlob(manifestBlob, `${downloadModelName}.manifest.json`);
  }
}
