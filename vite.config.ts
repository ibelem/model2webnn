import { defineConfig } from 'vite';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  plugins: [],
  base: '/model2webnn/',
  root: 'src/web',
  publicDir: '../../public',
  build: {
    outDir: '../../dist',
    emptyOutDir: true,
    rollupOptions: {
      onwarn(warning, warn) {
        // Suppress eval warning from protobufjs
        if (warning.code === 'EVAL' && warning.id?.includes('@protobufjs')) return;
        warn(warning);
      },
    },
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
