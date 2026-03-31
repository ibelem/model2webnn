// Preview module — Monaco editor for code preview with tab switching
// Manages grouped tabs: Code (JS, HTML), Inspect (Manifest, Weight Reader, Op Mapping), Run (Preview)

import type { ConvertResult } from '../index.js';
import * as monaco from 'monaco-editor';
import { initReader } from './reader.js';
import { initMapping } from './mapping.js';
import { initRunner } from './runner.js';

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

// Tabs that use the Monaco editor vs custom panels
const editorTabs = new Set(['js', 'html', 'manifest']);
const panelIds: Record<string, string> = {
  reader: 'readerPanel',
  mapping: 'mappingPanel',
  preview: 'previewPanel',
};

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

  // Reset to JS tab — reset both panel visibility and tab button active states
  const allTabs = document.querySelectorAll('.tab-bar .tab');
  allTabs.forEach((t) => t.classList.remove('active'));
  const jsTab = document.querySelector('.tab-bar .tab[data-tab="js"]');
  if (jsTab) jsTab.classList.add('active');
  showPanel('js');

  // Initialize sub-modules (reader/mapping don't attach to tab bar elements)
  initReader(result);
  initMapping(result);

  // Tab switching — use event delegation on the tab bar
  const tabBar = document.querySelector('.tab-bar')!;
  // Remove old listener by cloning
  const newTabBar = tabBar.cloneNode(true) as HTMLElement;
  tabBar.parentNode!.replaceChild(newTabBar, tabBar);

  // Initialize runner AFTER tab bar clone so its event listeners survive
  initRunner(result);

  newTabBar.addEventListener('click', (e) => {
    const target = (e.target as HTMLElement).closest('.tab') as HTMLElement | null;
    if (!target) return;
    const tab = target.dataset.tab;
    if (!tab) return;

    // Update active states
    newTabBar.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    target.classList.add('active');

    showPanel(tab);
  });

  // Expand/collapse toggle — sidebar visibility
  const toggleBtn = newTabBar.querySelector('#tabExpandToggle') as HTMLElement | null;
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const mainEl = document.querySelector('main');
      if (!mainEl) return;
      const isExpanded = mainEl.classList.toggle('full-content');
      toggleBtn.classList.toggle('active', isExpanded);
      toggleBtn.title = isExpanded ? 'Show sidebar' : 'Expand to full width';
    });
  }
}

function showPanel(tab: string): void {
  const editorContainer = document.getElementById('editorContainer')!;
  const previewControls = document.getElementById('previewControls');

  // Hide all custom panels
  for (const id of Object.values(panelIds)) {
    document.getElementById(id)!.style.display = 'none';
  }

  // Show/hide preview controls in the tab bar
  if (previewControls) {
    previewControls.style.display = tab === 'preview' ? '' : 'none';
  }

  if (editorTabs.has(tab)) {
    // Show Monaco editor
    editorContainer.style.display = '';
    switchEditorContent(tab);
  } else {
    // Hide Monaco, show the custom panel
    editorContainer.style.display = 'none';
    const panelId = panelIds[tab];
    if (panelId) {
      document.getElementById(panelId)!.style.display = '';
    }
  }
}

function switchEditorContent(tab: string): void {
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
