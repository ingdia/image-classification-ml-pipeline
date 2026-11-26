import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, /api is proxied to the nginx load balancer (docker compose on :8080).
// In production the built files are served by nginx itself, so /api is same-origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8080" },
  },
});
