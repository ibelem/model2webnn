// CLI entry point — converts model files from the command line
// Usage: npx tsx src/cli.ts <model.onnx> [--output dir] [--format js|ts|html]

import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { basename, extname, join, resolve, dirname } from 'node:path';
import { convert, getSupportedOnnxOps, getSupportedTfliteOps, getExternalDataRefs, getFreeDimensions, type ExternalDataMap } from './index.js';

interface CliArgs {
  inputPath: string;
  outputDir: string;
  format: 'javascript' | 'typescript' | 'html';
  listOps: boolean;
  help: boolean;
  freeDims: Record<string, number>;
}

function parseArgs(argv: string[]): CliArgs {
  const args = argv.slice(2);
  const result: CliArgs = {
    inputPath: '',
    outputDir: '.',
    format: 'javascript',
    listOps: false,
    help: false,
    freeDims: {},
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--help' || arg === '-h') {
      result.help = true;
    } else if (arg === '--list-ops') {
      result.listOps = true;
    } else if (arg === '--output' || arg === '-o') {
      result.outputDir = args[++i] ?? '.';
    } else if (arg === '--format' || arg === '-f') {
      const fmt = args[++i] ?? 'js';
      if (fmt === 'ts' || fmt === 'typescript') result.format = 'typescript';
      else if (fmt === 'html') result.format = 'html';
      else result.format = 'javascript';
    } else if (arg === '--free-dim' || arg === '-d') {
      const val = args[++i] ?? '';
      const eqIdx = val.indexOf('=');
      if (eqIdx > 0) {
        const name = val.substring(0, eqIdx);
        const num = parseInt(val.substring(eqIdx + 1), 10);
        if (!isNaN(num) && num > 0) {
          result.freeDims[name] = num;
        } else {
          console.error(`Invalid free dimension override: ${val} (value must be a positive integer)`);
          process.exit(1);
        }
      } else {
        console.error(`Invalid free dimension override format: ${val} (expected name=value)`);
        process.exit(1);
      }
    } else if (!arg.startsWith('-')) {
      result.inputPath = arg;
    }
  }

  return result;
}

function printUsage(): void {
  console.log(`
model2webnn — Convert ML models to WebNN JavaScript code

Usage:
  npx tsx src/cli.ts <model.onnx> [options]

Options:
  -o, --output <dir>     Output directory (default: current directory)
  -f, --format <fmt>     Output format: js, ts, html (default: js)
  -d, --free-dim <n=v>   Override a free dimension (e.g. batch_size=1)
                         Can be specified multiple times
  --list-ops             List supported operations
  -h, --help             Show this help message

Examples:
  npx tsx src/cli.ts model.onnx
  npx tsx src/cli.ts model.onnx -o output/ -f html
  npx tsx src/cli.ts model.onnx -d batch_size=1 -d sequence_length=128
  npx tsx src/cli.ts --list-ops
`);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv);

  if (args.help) {
    printUsage();
    return;
  }

  if (args.listOps) {
    const onnxOps = getSupportedOnnxOps();
    console.log(`Supported ONNX operations (${onnxOps.length}):`);
    for (const op of onnxOps.sort()) {
      console.log(`  ${op}`);
    }
    const tfliteOps = getSupportedTfliteOps();
    console.log(`\nSupported TFLite operations (${tfliteOps.length}):`);
    for (const op of tfliteOps.sort()) {
      console.log(`  ${op}`);
    }
    return;
  }

  if (!args.inputPath) {
    console.error('Error: No input file specified.\n');
    printUsage();
    process.exit(1);
  }

  const inputPath = resolve(args.inputPath);
  const modelName = basename(inputPath, extname(inputPath));
  const outputDir = resolve(args.outputDir);

  console.log(`Converting: ${inputPath}`);
  console.log(`Output:     ${outputDir}/`);
  console.log(`Format:     ${args.format}`);
  console.log();

  // Read model file
  const buffer = new Uint8Array(await readFile(inputPath));
  console.log(`Model size: ${(buffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

  // Detect and load external data files
  const externalData = await loadExternalData(inputPath, buffer);

  // Convert
  const weightsFileName = `${modelName}.weights`;
  const manifestFileName = `${modelName}.manifest.json`;

  // Detect free dimensions and merge with CLI overrides
  const freeDimensionOverrides = args.freeDims;

  // Quick-parse to detect free dimensions and warn about unresolved ones
  const modelFormat = buffer[0] === 0x08 || buffer[0] === 0x3a || buffer[0] === 0x12 ? 'onnx' : 'tflite';
  if (modelFormat === 'onnx') {
    const { parseOnnx } = await import('./parsers/onnx.js');
    const tempGraph = await parseOnnx(buffer, externalData.size > 0 ? externalData : undefined);
    const freeDims = getFreeDimensions(tempGraph);
    if (freeDims.length > 0) {
      const resolved = freeDims.filter((d) => d in freeDimensionOverrides);
      const unresolved = freeDims.filter((d) => !(d in freeDimensionOverrides));
      if (resolved.length > 0) {
        console.log(`Free dimensions overridden: ${resolved.map((d) => `${d}=${freeDimensionOverrides[d]}`).join(', ')}`);
      }
      if (unresolved.length > 0) {
        console.warn(`Warning: Unresolved free dimensions: ${unresolved.join(', ')}`);
        console.warn(`  Use --free-dim <name>=<value> to set fixed values.`);
        console.warn(`  See: https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#freedimensionoverrides`);
      }
    }
  }

  const result = await convert(buffer, {
    format: args.format,
    weightsFileName,
    manifestFileName,
    modelName,
    externalData: externalData.size > 0 ? externalData : undefined,
    freeDimensionOverrides: Object.keys(freeDimensionOverrides).length > 0 ? freeDimensionOverrides : undefined,
  });

  // Write outputs
  await mkdir(outputDir, { recursive: true });

  const ext = args.format === 'typescript' ? '.ts' : '.js';
  const codePath = join(outputDir, `${modelName}${ext}`);
  await writeFile(codePath, result.code, 'utf-8');
  console.log(`Code:       ${codePath}`);

  const weightsPath = join(outputDir, weightsFileName);
  await writeFile(weightsPath, result.weights);
  console.log(`Weights:    ${weightsPath} (${(result.weights.byteLength / 1024 / 1024).toFixed(2)} MB)`);

  const manifestPath = join(outputDir, manifestFileName);
  await writeFile(manifestPath, JSON.stringify(result.manifest, null, 2), 'utf-8');
  console.log(`Manifest:   ${manifestPath}`);

  if (result.html) {
    const htmlPath = join(outputDir, `${modelName}.html`);
    await writeFile(htmlPath, result.html, 'utf-8');
    console.log(`HTML:       ${htmlPath}`);
  }

  // Print graph summary
  const g = result.graph;
  console.log();
  console.log(`Graph: ${g.name}`);
  console.log(`  Inputs:    ${g.inputs.length}`);
  console.log(`  Outputs:   ${g.outputs.length}`);
  console.log(`  Constants: ${g.constants.length}`);
  console.log(`  Nodes:     ${g.nodes.length}`);

  // Operator coverage report
  const cov = result.coverage;
  console.log(`  Coverage:  ${cov.coveragePercent}% (${cov.supportedOps}/${cov.totalOps} ops supported)`);

  if (cov.unsupportedOps > 0) {
    console.log();
    console.warn(`  Warning: ${cov.unsupportedOps} unsupported op(s) found:`);
    for (const { opType, count } of cov.unsupportedOpTypes) {
      console.warn(`    ${opType} (×${count})`);
    }
    console.warn(`  Generated code will contain placeholder stubs for unsupported ops.`);
  }

  console.log('\nDone.');
}

/**
 * Detect and load ONNX external data files.
 * Strategy:
 *   1. Parse the model to find external_data references (file paths)
 *   2. Load each referenced file from the same directory as the .onnx file
 *   3. If no references found, scan for common patterns (model.onnx_data, etc.)
 */
async function loadExternalData(
  modelPath: string,
  modelBuffer: Uint8Array,
): Promise<ExternalDataMap> {
  const externalData: ExternalDataMap = new Map();
  const modelDir = dirname(modelPath);

  // First, scan the model proto for external data references
  const refs = getExternalDataRefs(modelBuffer);

  if (refs.length > 0) {
    console.log(`External data: ${refs.length} file(s) referenced`);
    for (const ref of refs) {
      const filePath = join(modelDir, ref);
      try {
        const data = new Uint8Array(await readFile(filePath));
        externalData.set(ref, data);
        console.log(`  Loaded: ${ref} (${(data.byteLength / 1024 / 1024).toFixed(2)} MB)`);
      } catch {
        throw new Error(
          `External data file not found: ${filePath}\n` +
          `  Ensure the file is in the same directory as the model.`
        );
      }
    }
    return externalData;
  }

  // Fallback: auto-detect common external data file patterns
  // e.g. model.onnx_data, model.onnx_data_1, model.onnx_data_2, ...
  const modelFileName = basename(modelPath);
  try {
    const dirEntries = await readdir(modelDir);
    const externalFiles = dirEntries.filter((f) => {
      // Match: model.onnx_data, model.onnx_data_1, model.onnx.data, etc.
      return f.startsWith(modelFileName) && f !== modelFileName;
    });

    if (externalFiles.length > 0) {
      console.log(`External data: ${externalFiles.length} file(s) auto-detected`);
      for (const fileName of externalFiles.sort()) {
        const filePath = join(modelDir, fileName);
        const data = new Uint8Array(await readFile(filePath));
        externalData.set(fileName, data);
        console.log(`  Loaded: ${fileName} (${(data.byteLength / 1024 / 1024).toFixed(2)} MB)`);
      }
    }
  } catch {
    // Directory read failed, skip auto-detection
  }

  return externalData;
}

main().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});
