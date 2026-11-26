import { useEffect, useState } from "react";
import { api, formatUptime } from "../api.js";

// Model uptime monitoring: polls /health every 5 seconds.
export default function StatusBar() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(false);
  const [checkedAt, setCheckedAt] = useState(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await api.health();
        if (alive) { setHealth(h); setError(false); }
      } catch {
        if (alive) setError(true);
      }
      if (alive) setCheckedAt(new Date());
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  if (error) return <div className="status"><span className="pill bad">API offline</span></div>;
  if (!health) return <div className="status"><span className="pill">Connecting…</span></div>;
  return (
    <div className="status" title={checkedAt ? `Last checked ${checkedAt.toLocaleTimeString()}` : ""}>
      <span className={`pill ${health.status === "ok" ? "ok" : "warn"}`}>
        {health.status === "ok" ? "Online" : "Degraded"}
      </span>
      <span className="pill">Uptime {formatUptime(health.uptime_seconds)}</span>
      <span className="pill">Model v{health.model_version ?? "–"}</span>
      {health.retraining && <span className="pill warn">Retraining…</span>}
      <span className="pill" title="Container that answered (changes when scaled)">Container {health.container?.slice(0, 6)}</span>
    </div>
  );
}
