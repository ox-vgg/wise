import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    proxy: {
      '/media/': process.env.API_BASE_URL,
      '/thumbnail': process.env.API_BASE_URL,
      '/metadata/': process.env.API_BASE_URL,
      '/storyboard/': process.env.API_BASE_URL,
      '/info': process.env.API_BASE_URL,
      '/report': process.env.API_BASE_URL,
      '/featured': process.env.API_BASE_URL,
      '/related-vectors/': process.env.API_BASE_URL,
      '/search': process.env.API_BASE_URL,
      '/shard/': process.env.API_BASE_URL,
      '/api/facets': process.env.API_BASE_URL,
    }
  },
  plugins: [react()],
  base: './',
  build: {
    manifest: true,
    rollupOptions: {
      external: ["./config.js"],
      input: {
        main: './index.html',
        facets: './facets.html'
      }
    }
  },
})
