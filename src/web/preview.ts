// Preview module — Monaco editor for code preview with tab switching

import type { ConvertResult } from '../index.js';
import * as monaco from 'monaco-editor';

import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import jsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
import cssWorker from 'monaco-editor/esm/vs/language/css/css.worker?worker';
import htmlWorker from 'monaco-editor/esm/vs/language/html/html.worker?worker';
import tsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker';

self.MonacoEnvironment = {
  getWorker(_: unknown, label: string) {
    if (label === 'json') return new jsonWorker();
    if (label === 'css' || label === 'scss' || label === 'less') return new cssWorker();
    if (label === 'html' || label === 'handlebars' || label === 'razor') return new htmlWorker();
    if (label === 'typescript' || label === 'javascript') return new tsWorker();
    return new editorWorker();
  },
};

let editor: monaco.editor.IStandaloneCodeEditor | null = null;
let currentResult: ConvertResult | null = null;

export function initPreview(result: ConvertResult): void {
  currentResult = result;

  // Create editor if not exists
  const container = document.getElementById('editorContainer')!;
  if (!editor) {
    editor = monaco.editor.create(container, {
      value: result.code,
      language: 'javascript',
      readOnly: true,
      theme: 'vs',
      minimap: { enabled: false },
      fontSize: 13,
      lineNumbers: 'on',
      scrollBeyondLastLine: false,
      automaticLayout: true,
      wordWrap: 'on',
    });
  } else {
    editor.setValue(result.code);
    monaco.editor.setModelLanguage(editor.getModel()!, 'javascript');
  }

  // Tab switching
  const tabs = document.querySelectorAll<HTMLElement>('.tab');
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      switchTab(tab.dataset.tab ?? 'js');
    });
  });
}

function switchTab(tab: string): void {
  if (!editor || !currentResult) return;

  switch (tab) {
    case 'js':
      editor.setValue(currentResult.code);
      monaco.editor.setModelLanguage(editor.getModel()!, 'javascript');
      break;
    case 'html':
      editor.setValue(currentResult.html ?? '<!-- No HTML output -->');
      monaco.editor.setModelLanguage(editor.getModel()!, 'html');
      break;
    case 'manifest':
      editor.setValue(JSON.stringify(currentResult.manifest, null, 2));
      monaco.editor.setModelLanguage(editor.getModel()!, 'json');
      break;
  }
}
