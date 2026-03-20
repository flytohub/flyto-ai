import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import UnoCSS from 'unocss/vite'

export default defineConfig({
  base: '/console/',
  plugins: [vue(), UnoCSS()],
  build: {
    outDir: '../flyto_ai/console/dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/console/api': 'http://localhost:7411',
    },
  },
})
