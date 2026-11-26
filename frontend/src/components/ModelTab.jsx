import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, pct } from "../api.js";

function ConfusionMatrix({ cm, classes }) {
  const max = Math.max(...cm.flat());
  return (
    <div className="table-wrap">
      <table className="cm">
        <thead>
          <tr><th className="rowh">actual ↓ / predicted →</th>{classes.map((c) => <th key={c}>{c.slice(0, 5)}</th>)}</tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={i}>
              <th className="rowh">{classes[i]}</th>
              {row.map((v, j) => {
                const a = v / max;
                const bg = i === j ? `rgba(26,127,69,${0.12 + 0.75 * a})` : v ? `rgba(180,35,24,${0.1 + 0.7 * a})` : "transparent";
                return <td key={j} style={{ background: bg, color: a > 0.6 ? "#fff" : "inherit" }}>{v}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricsBlock({ m }) {
  return (
    <div className="kpis">
      <div className="kpi"><div className="label">Accuracy</div><div className="value">{pct(m.accuracy)}</div></div>
      <div className="kpi"><div className="label">Precision (macro)</div><div className="value">{pct(m.precision_macro)}</div></div>
      <div className="kpi"><div className="label">Recall (macro)</div><div className="value">{pct(m.recall_macro)}</div></div>
      <div className="kpi"><div className="label">F1-score (macro)</div><div className="value">{pct(m.f1_macro)}</div></div>
    </div>
  );
}

// Model performance: offline test metrics, training curves, version history and live evaluation.
export default function ModelTab() {
  const [data, setData] = useState(null);
  const [live, setLive] = useState(null);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => { api.metrics().then(setData).catch((e) => setError(e.message)); }, []);

  const runEval = async () => {
    setEvaluating(true); setError(null);
    try { setLive(await api.evaluate()); } catch (e) { setError(e.message); }
    finally { setEvaluating(false); }
  };

  if (error && !data) return <div className="error">{error}</div>;
  if (!data) return <p className="muted">Loading model metrics…</p>;
  const m = data.test_metrics;
  const h = data.training_history;
  const curve = h ? h.accuracy.map((_, i) => ({
    epoch: i + 1, train: +(h.accuracy[i] * 100).toFixed(1), validation: +(h.val_accuracy[i] * 100).toFixed(1),
  })) : [];
  const report = m?.classification_report;

  return (
    <div className="stack">
      <section className="card stack">
        <h2>Live model v{data.current?.version} — test-set performance</h2>
        <p className="muted">Measured on {m?.n_samples} held-out test documents that were never used for training (random guessing ≈ 11.1%).</p>
        {m && <MetricsBlock m={m} />}
      </section>

      <section className="card stack">
        <h2>Evaluation in production</h2>
        <p className="muted">
          Re-scores the model that is serving predictions <em>right now</em> on the test set, inside the running
          container. After a retraining is promoted, this confirms the new live model's real performance.
        </p>
        <div className="row">
          <button className="primary" onClick={runEval} disabled={evaluating}>
            {evaluating ? "Evaluating… (about 20–60 s)" : "Evaluate live model now"}
          </button>
        </div>
        {error && <div className="error">{error}</div>}
        {live && (
          <div className="stack">
            <div className="notice">
              Model v{live.model_version} evaluated on {live.n_samples} documents in {live.evaluation_seconds} s
              by container {live.container?.slice(0, 6)} at {new Date(live.evaluated_at).toLocaleTimeString()}.
            </div>
            <MetricsBlock m={live} />
          </div>
        )}
      </section>

      <div className="grid">
        <section className="card">
          <h2>Confusion matrix</h2>
          {m && <ConfusionMatrix cm={(live || m).confusion_matrix} classes={m.class_names} />}
          <p className="muted">Green diagonal = correct predictions. Red cells = documents of the row type mistaken for the column type.</p>
        </section>
        <section className="card">
          <h2>Per-type results</h2>
          {report && (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Type</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
                <tbody>
                  {m.class_names.map((c) => (
                    <tr key={c}>
                      <td>{c}</td><td>{report[c].precision.toFixed(2)}</td><td>{report[c].recall.toFixed(2)}</td>
                      <td>{report[c]["f1-score"].toFixed(2)}</td><td>{report[c].support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      {h && (
        <section className="card">
          <h2>Training curves (initial model)</h2>
          <div style={{ width: "100%", height: 280 }}>
            <ResponsiveContainer>
              <LineChart data={curve}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis unit="%" domain={[0, 100]} />
                <Tooltip />
                <Legend />
                <ReferenceLine x={h.fine_tune_start_epoch + 0.5} stroke="#999" strokeDasharray="4 4" label="fine-tuning" />
                <Line dataKey="train" stroke="#1f5fbf" dot={false} />
                <Line dataKey="validation" stroke="#e07b39" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <section className="card">
        <h2>Model version history</h2>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Version</th><th>Kind</th><th>Status</th><th>Accuracy</th><th>F1</th><th>Train images</th><th>Created</th></tr></thead>
            <tbody>
              {[...data.history].reverse().map((v) => (
                <tr key={v.version}>
                  <td>v{v.version}{v.version === data.current?.version ? " (live)" : ""}</td>
                  <td>{v.kind}</td>
                  <td><span className={`badge ${v.status}`}>{v.status}</span></td>
                  <td>{pct(v.metrics?.accuracy)}</td>
                  <td>{pct(v.metrics?.f1_macro)}</td>
                  <td>{v.train_samples}{v.new_samples ? ` (+${v.new_samples} new)` : ""}</td>
                  <td>{v.created_at ? new Date(v.created_at).toLocaleString() : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
