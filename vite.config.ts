import { defineConfig } from 'vite';
import monacoEditorEsmPlugin from 'vite-plugin-monaco-editor-esm';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  plugins: [monacoEditorEsmPlugin()],
  base: '/model2webnn/',
  root: 'src/web',
  publicDir: '../../public',
  build: {
    outDir: '../../dist',
    emptyOutDir: true,
  },
  server: {
    fs: {
      allow: ['../..'],
    },
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
});
