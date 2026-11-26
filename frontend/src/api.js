// Thin wrapper around the FastAPI backend. All requests go to /api/... which nginx
// forwards (round-robin) to one of the API containers.
const BASE = (import.meta.env.VITE_API_URL || "") + "/api";

async function request(path, options = {}) {
  const res = await fetch(BASE + path, options);
  const isJson = res.headers.get("content-type")?.includes("application/json");
  const body = isJson ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = body?.detail;
    const message = typeof detail === "string" ? detail : detail?.message || res.statusText;
    const err = new Error(`${res.status}: ${message}`);
    err.body = body;
    throw err;
  }
  return body;
}

export const api = {
  health: () => request("/health"),
  metrics: () => request("/metrics"),
  datasetStats: () => request("/dataset/stats"),
  samples: () => request("/samples"),
  sampleUrl: (name) => `${BASE}/samples/${encodeURIComponent(name)}`,
  predict: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/predict", { method: "POST", body: form });
  },
  upload: (files, label) => {
    const form = new FormData();
    [...files].forEach((f) => form.append("files", f));
    if (label) form.append("label", label);
    return request("/upload", { method: "POST", body: form });
  },
  uploads: () => request("/uploads"),
  clearUploads: () => request("/uploads", { method: "DELETE" }),
  retrain: () => request("/retrain", { method: "POST" }),
  retrainStatus: () => request("/retrain/status"),
  evaluate: () => request("/evaluate", { method: "POST" }),
};

export const pct = (x, digits = 1) => (x == null ? "–" : `${(x * 100).toFixed(digits)}%`);

export function formatUptime(seconds) {
  if (seconds == null) return "–";
  const s = Math.floor(seconds);
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return [d && `${d}d`, (d || h) && `${h}h`, `${m}m`, `${s % 60}s`].filter(Boolean).join(" ");
}
