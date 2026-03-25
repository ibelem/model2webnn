// Weight extraction and WGWT binary packing
// Produces .weights (binary) + .manifest.json from GraphIR constants.

import type { ConstantInfo, MLOperandDataType } from '../ir/graph.js';

export interface WeightsManifest {
  format: 'wg-weights-manifest';
  version: 1;
  endianness: 'little';
  tensors: Record<string, TensorManifestEntry>;
}

export interface TensorManifestEntry {
  dataType: MLOperandDataType;
  shape: number[];
  byteOffset: number;
  byteLength: number;
}

// WGWT header: "WGWT" magic (4 bytes) + version 1 (4 bytes, little-endian u32)
const WGWT_MAGIC = new Uint8Array([0x57, 0x47, 0x57, 0x54]); // "WGWT"
const WGWT_VERSION = 1;
const HEADER_SIZE = 8;

export interface PackedWeights {
  weights: Uint8Array; // WGWT binary file
  manifest: WeightsManifest; // manifest.json content
}

export function packWeights(constants: ConstantInfo[]): PackedWeights {
  // Calculate total size: header + all tensor data
  let totalDataSize = 0;
  for (const c of constants) {
    totalDataSize += c.rawData.byteLength;
  }

  const buffer = new Uint8Array(HEADER_SIZE + totalDataSize);

  // Write header
  buffer.set(WGWT_MAGIC, 0);
  const view = new DataView(buffer.buffer);
  view.setUint32(4, WGWT_VERSION, true); // little-endian

  // Pack tensors and build manifest
  const tensors: Record<string, TensorManifestEntry> = {};
  let offset = HEADER_SIZE;

  for (const c of constants) {
    buffer.set(c.rawData, offset);

    // Only include numeric shapes in manifest (filter out dynamic dims)
    const numericShape = c.shape.map((d) =>
      typeof d === 'number' ? d : 0
    );

    tensors[c.name] = {
      dataType: c.dataType,
      shape: numericShape,
      byteOffset: offset,
      byteLength: c.rawData.byteLength,
    };

    // Update constant with packed offset
    c.byteOffset = offset;
    offset += c.rawData.byteLength;
  }

  const manifest: WeightsManifest = {
    format: 'wg-weights-manifest',
    version: 1,
    endianness: 'little',
    tensors,
  };

  return { weights: buffer, manifest };
}
