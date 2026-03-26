// Weight Reader module — interactive tensor viewer
// Left sidebar: tensor list, Right: typed values + hex dump

import type { ConvertResult } from '../index.js';
import { getTypedArrayName, bytesPerElement, type MLOperandDataType } from '../ir/graph.js';
import type { WeightsManifest } from '../weights/packer.js';

let currentResult: ConvertResult | null = null;

const MAX_DISPLAY_VALUES = 500;
const HEX_BYTES_PER_ROW = 16;
const MAX_HEX_BYTES = 4096;

export function initReader(result: ConvertResult): void {
  currentResult = result;

  const sidebar = document.getElementById('readerSidebar')!;
  const stringEl = document.getElementById('readerString')!;
  const bufferEl = document.getElementById('readerBuffer')!;

  // Build sidebar tensor list
  sidebar.innerHTML = '';
  const manifest = result.manifest;
  const tensorNames = Object.keys(manifest.tensors);

  if (tensorNames.length === 0) {
    sidebar.innerHTML = '<div class="reader-empty">No tensors</div>';
    stringEl.textContent = '';
    bufferEl.textContent = '';
    return;
  }

  for (const name of tensorNames) {
    const info = manifest.tensors[name];
    const btn = document.createElement('button');
    btn.className = 'reader-sidebar-item';
    btn.title = name;
    btn.dataset.tensor = name;

    const sizeKB = (info.byteLength / 1024).toFixed(1);
    btn.innerHTML =
      `${escapeHtml(name)}<span class="reader-sidebar-meta">${info.dataType} ${JSON.stringify(info.shape)} · ${sizeKB} KB</span>`;

    btn.addEventListener('click', () => selectTensor(name));
    sidebar.appendChild(btn);
  }

  // Select first tensor by default
  stringEl.innerHTML = '<div class="reader-empty">Select a tensor from the sidebar</div>';
  bufferEl.textContent = '';
}

function selectTensor(name: string): void {
  if (!currentResult) return;


  // Update sidebar active state
  const sidebar = document.getElementById('readerSidebar')!;
  sidebar.querySelectorAll('.reader-sidebar-item').forEach((btn) => {
    btn.classList.toggle('active', (btn as HTMLElement).dataset.tensor === name);
  });

  const manifest = currentResult.manifest;
  const info = manifest.tensors[name];
  if (!info) return;

  const weightsBuffer = currentResult.weights.buffer as ArrayBuffer;
  const { byteOffset, byteLength } = info;

  renderTypedData(info, weightsBuffer, byteOffset, byteLength);
  renderHexDump(weightsBuffer, byteOffset, byteLength);
}

function renderTypedData(
  info: WeightsManifest['tensors'][string],
  buffer: ArrayBuffer,
  byteOffset: number,
  byteLength: number,
): void {
  const el = document.getElementById('readerString')!;
  const dataType = info.dataType as MLOperandDataType;
  const arrayName = getTypedArrayName(dataType);
  const bpe = bytesPerElement(dataType);
  const count = Math.floor(byteLength / bpe);

  let values: string;
  try {
    const TypedArrayCtor = getTypedArrayCtor(info.dataType);
    // Copy to an aligned buffer — TypedArrays require alignment matching their
    // element size (e.g. Float32Array needs 4-byte alignment). WGWT packs
    // tensors sequentially without padding, so offsets are often unaligned.
    const aligned = new ArrayBuffer(byteLength);
    new Uint8Array(aligned).set(new Uint8Array(buffer, byteOffset, byteLength));
    const arr = new TypedArrayCtor(aligned, 0, count);
    const displayCount = Math.min(count, MAX_DISPLAY_VALUES);
    const parts: string[] = [];
    for (let i = 0; i < displayCount; i++) {
      parts.push(String(arr[i]));
    }
    values = parts.join(', ');
    if (count > MAX_DISPLAY_VALUES) {
      values += `\n\n... (${count - MAX_DISPLAY_VALUES} more values, ${count} total)`;
    }
  } catch {
    values = `[Unable to decode as ${arrayName}]`;
  }

  el.textContent = `${arrayName}[${count}]  shape: ${JSON.stringify(info.shape)}\n\n${values}`;
}

function renderHexDump(buffer: ArrayBuffer, byteOffset: number, byteLength: number): void {
  const el = document.getElementById('readerBuffer')!;
  const bytes = new Uint8Array(buffer, byteOffset, byteLength);
  const displayLen = Math.min(byteLength, MAX_HEX_BYTES);

  const lines: string[] = [];
  for (let i = 0; i < displayLen; i += HEX_BYTES_PER_ROW) {
    const offset = (byteOffset + i).toString(16).padStart(8, '0');
    const hexParts: string[] = [];
    let ascii = '';

    for (let j = 0; j < HEX_BYTES_PER_ROW; j++) {
      if (i + j < displayLen) {
        const b = bytes[i + j];
        hexParts.push(b.toString(16).padStart(2, '0'));
        ascii += (b >= 0x20 && b <= 0x7e) ? String.fromCharCode(b) : '.';
      } else {
        hexParts.push('  ');
        ascii += ' ';
      }
    }

    lines.push(`${offset}  ${hexParts.join(' ')}  |${ascii}|`);
  }

  if (byteLength > MAX_HEX_BYTES) {
    lines.push(`\n... (${byteLength - MAX_HEX_BYTES} more bytes, ${byteLength} total)`);
  }

  el.textContent = lines.join('\n');
}

function getTypedArrayCtor(dataType: string): any {
  switch (dataType) {
    case 'float32': return Float32Array;
    case 'float16': return Uint16Array; // Float16Array may not be available
    case 'int64': return BigInt64Array;
    case 'uint64': return BigUint64Array;
    case 'int32': return Int32Array;
    case 'uint32': return Uint32Array;
    case 'int8': return Int8Array;
    case 'uint8': return Uint8Array;
    default: return Uint8Array;
  }
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
