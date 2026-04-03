// WebNN DSL code generator — emits .webnn text format
// Produces a human-readable graph representation compatible with the
// webnn-graph DSL proposal (https://github.com/webmachinelearning/proposals/issues/16).

import type { GraphIR, MLOperandDataType } from '../ir/graph.js';

export interface GenerateWebnnDslOptions {
  /** Include original model op type as comments (default: true) */
  includeComments?: boolean;
}

/** Map MLOperandDataType → DSL dtype token */
function dslDtype(dt: MLOperandDataType): string {
  switch (dt) {
    case 'float32': return 'f32';
    case 'float16': return 'f16';
    case 'int64': return 'i64';
    case 'uint64': return 'u64';
    case 'int32': return 'i32';
    case 'uint32': return 'u32';
    case 'int8': return 'i8';
    case 'uint8': return 'u8';
  }
}

/** Escape a string for the DSL (double-quoted) */
function dslString(s: string): string {
  return `"${s.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
}

/** Convert a tensor name to a valid DSL identifier (ASCII_ALPHA | _ | / | $) start */
function dslIdent(name: string): string {
  // Replace characters not allowed in the grammar's ident rule
  let id = name.replace(/[^a-zA-Z0-9_./$/]/g, '_');
  // Ensure starts with valid char (alpha, _, /, $)
  if (id.length === 0 || /^[0-9.]/.test(id)) {
    id = '_' + id;
  }
  return id;
}

/** Format a shape dimension for the DSL */
function dslDim(dim: number | string): string {
  if (typeof dim === 'number') return dim.toString();
  // Dynamic dimension — use dyn("name", 1) with default max 1
  return `dyn(${dslString(dim)}, 1)`;
}

/** Serialize an attribute value to DSL syntax */
function dslValue(val: unknown): string {
  if (val === null || val === undefined) return 'null';
  if (typeof val === 'boolean') return val ? 'true' : 'false';
  if (typeof val === 'number') return Object.is(val, -0) ? '-0' : val.toString();
  if (typeof val === 'string') return dslString(val);
  if (Array.isArray(val)) {
    return `[${val.map(dslValue).join(', ')}]`;
  }
  if (typeof val === 'object') {
    const entries = Object.entries(val as Record<string, unknown>);
    const kvs = entries.map(([k, v]) => `${k}: ${dslValue(v)}`);
    return `{${kvs.join(', ')}}`;
  }
  return String(val);
}

/**
 * Generate a .webnn DSL text representation of a GraphIR.
 */
export function generateWebnnDsl(
  graph: GraphIR,
  options: GenerateWebnnDslOptions = {},
): string {
  const { includeComments = true } = options;

  const lines: string[] = [];
  const I = '  ';    // indent level 1
  const II = '    '; // indent level 2

  // Header
  lines.push(`webnn_graph ${dslString(graph.name)} v1 {`);

  // Inputs block
  lines.push(`${I}inputs {`);
  for (const inp of graph.inputs) {
    const shape = inp.shape.map(dslDim).join(', ');
    lines.push(`${II}${dslIdent(inp.name)}: ${dslDtype(inp.dataType)}[${shape}];`);
  }
  lines.push(`${I}}`);
  lines.push('');

  // Consts block
  if (graph.constants.length > 0) {
    lines.push(`${I}consts {`);
    for (const c of graph.constants) {
      const shape = c.shape.map(dslDim).join(', ');
      lines.push(`${II}${dslIdent(c.name)}: ${dslDtype(c.dataType)}[${shape}] @weights(${dslString(c.name)});`);
    }
    lines.push(`${I}}`);
    lines.push('');
  }

  // Nodes block
  lines.push(`${I}nodes {`);
  for (const node of graph.nodes) {
    // Optional comment with original op type and format
    if (includeComments) {
      lines.push(`${II}# ${node.opType} (${graph.format})`);
    }

    // Positional args: input tensor references
    const posArgs = node.inputs
      .filter((name) => name !== '')
      .map(dslIdent);

    // Named args: attributes
    const namedArgs: string[] = [];
    for (const [key, val] of Object.entries(node.attributes)) {
      if (val === undefined) continue;
      namedArgs.push(`${key}=${dslValue(val)}`);
    }

    const allArgs = [...posArgs, ...namedArgs].join(', ');

    // Output assignment
    const outputs = node.outputs.filter((name) => name !== '');
    if (outputs.length === 0) {
      // No outputs — just a call (shouldn't normally happen)
      lines.push(`${II}${node.opType}(${allArgs});`);
    } else if (outputs.length === 1) {
      lines.push(`${II}${dslIdent(outputs[0])} = ${node.opType}(${allArgs});`);
    } else {
      // Multi-output: [out1, out2] = op(...)
      const outIdents = outputs.map(dslIdent).join(', ');
      lines.push(`${II}[${outIdents}] = ${node.opType}(${allArgs});`);
    }
  }
  lines.push(`${I}}`);
  lines.push('');

  // Outputs block
  const outputIdents = graph.outputs.map((o) => dslIdent(o.name));
  if (outputIdents.length <= 3) {
    lines.push(`${I}outputs { ${outputIdents.join(', ')}; }`);
  } else {
    lines.push(`${I}outputs {`);
    for (const id of outputIdents) {
      lines.push(`${II}${id};`);
    }
    lines.push(`${I}}`);
  }

  lines.push('}');
  lines.push(''); // trailing newline

  return lines.join('\n');
}
