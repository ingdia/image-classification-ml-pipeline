import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { api } from "../api.js";

const COLORS = ["#1f5fbf", "#e07b39", "#2a9d8f", "#9b5de5", "#c0392b", "#6c757d", "#d4a017", "#3a86ff", "#8d6e63"];

// Dataset visualisations: 3 features of the scanned documents, each with an interpretation.
export default function DataTab() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => { api.datasetStats().then(setStats).catch((e) => setError(e.message)); }, []);

  if (error) return <div className="error">{error}</div>;
  if (!stats) return <p className="muted">Loading dataset statistics…</p>;

  const classes = stats.class_names;
  const dist = classes.map((c) => ({
    name: c, train: stats.distribution.train[c], val: stats.distribution.val[c], test: stats.distribution.test[c],
  }));
  const f = stats.per_class_features;
  const ink = classes
    .map((c) => ({ name: c, ink: +(f[c].ink_density.mean * 100).toFixed(2), brightness: +f[c].brightness.mean.toFixed(1) }))
    .sort((a, b) => b.ink - a.ink);
  const geometry = classes.map((c) => ({ name: c, width: +f[c].width.mean.toFixed(0), aspect: +f[c].aspect_ratio.mean.toFixed(3) }));
  const totals = { train: 0, val: 0, test: 0 };
  dist.forEach((d) => { totals.train += d.train; totals.val += d.val; totals.test += d.test; });
  const counts = dist.map((d) => d.train + d.val + d.test);
  const balanced = Math.max(...counts) === Math.min(...counts);
  const densest = ink[0], sparsest = ink[ink.length - 1];
  const widths = geometry.map((g) => g.width);

  return (
    <div className="stack">
      <div className="kpis">
        <div className="kpi"><div className="label">Total documents</div><div className="value">{stats.total_images}</div></div>
        <div className="kpi"><div className="label">Document types</div><div className="value">{classes.length}</div></div>
        <div className="kpi"><div className="label">Train / Val / Test</div><div className="value">{totals.train} / {totals.val} / {totals.test}</div></div>
        <div className="kpi"><div className="label">Source</div><div className="value" style={{ fontSize: "0.95rem" }}>RVL-CDIP small-200</div></div>
      </div>

      <section className="card">
        <h2>Feature 1 — Class distribution</h2>
        <div style={{ width: "100%", height: 300 }}>
          <ResponsiveContainer>
            <BarChart data={dist}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" interval={0} angle={-30} textAnchor="end" height={70} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="train" stackId="a" fill="#1f5fbf" />
              <Bar dataKey="val" stackId="a" fill="#7fa6e0" />
              <Bar dataKey="test" stackId="a" fill="#c9d8f0" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="insight">
          <strong>Story:</strong> {balanced
            ? `Every document type has exactly ${counts[0]} images (${dist[0].train} train / ${dist[0].val} validation / ${dist[0].test} test), so the dataset is perfectly balanced.`
            : "The dataset is imbalanced across types."}{" "}
          Accuracy is therefore a fair headline metric and the model cannot score well by always guessing a majority
          class. A real lender's inbox would be imbalanced (far more invoices and forms than resumes), so class
          weights would be needed in production. Uploaded retraining data can also unbalance the set — the
          validation step reports counts per type before retraining.
        </div>
      </section>

      <section className="card">
        <h2>Feature 2 — Page geometry (size and shape)</h2>
        <div className="grid">
          <div style={{ width: "100%", height: 280 }}>
            <ResponsiveContainer>
              <ScatterChart margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="width" name="width" unit="px" domain={["dataMin - 20", "dataMax + 20"]} />
                <YAxis type="number" dataKey="height" name="height" unit="px" domain={[900, 1100]} />
                <ZAxis range={[20, 20]} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                {classes.map((c, i) => (
                  <Scatter key={c} name={c} data={stats.sizes.filter((s) => s.class === c)} fill={COLORS[i]} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div style={{ width: "100%", height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={geometry}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" interval={0} angle={-30} textAnchor="end" height={70} />
                <YAxis domain={[1.2, 1.35]} />
                <Tooltip />
                <Bar dataKey="aspect" name="mean aspect ratio (h / w)" fill="#2a9d8f" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="insight">
          <strong>Story:</strong> Every scan is exactly 1000 px tall and the mean width per type only ranges from{" "}
          {Math.min(...widths)} to {Math.max(...widths)} px — all are portrait pages with an aspect ratio of about 1.3.
          Page <em>shape</em> carries almost no information about the document type, so the model must learn from
          the <em>layout inside</em> the page. Resizing every page to 224×224 squashes them all the same way, so it
          does not favour any type — but it does make small text unreadable, which limits accuracy.
        </div>
      </section>

      <section className="card">
        <h2>Feature 3 — Ink density and brightness</h2>
        <div style={{ width: "100%", height: 300 }}>
          <ResponsiveContainer>
            <BarChart data={ink}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" interval={0} angle={-30} textAnchor="end" height={70} />
              <YAxis yAxisId="ink" unit="%" />
              <YAxis yAxisId="b" orientation="right" domain={[220, 255]} />
              <Tooltip />
              <Legend />
              <Bar yAxisId="ink" dataKey="ink" name="ink density (% dark pixels)" fill="#1d2330" />
              <Bar yAxisId="b" dataKey="brightness" name="mean brightness (0–255)" fill="#b8c2d0" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="insight">
          <strong>Story:</strong> Pages are mostly white paper (brightness ≈ {Math.min(...ink.map((i) => i.brightness)).toFixed(0)}–
          {Math.max(...ink.map((i) => i.brightness)).toFixed(0)} out of 255), but the amount of ink differs by type.{" "}
          <strong>{densest.name}</strong> is the densest ({densest.ink}% dark pixels) while <strong>{sparsest.name}</strong> is
          by far the sparsest ({sparsest.ink}%) — short printed messages with lots of white space, which is why the
          model recognises emails well. Letters and memos have almost the same ink density and layout, and they are
          among the types the model confuses most. Ink alone cannot separate all 9 types, which is why a CNN that
          learns spatial layout is used.
        </div>
      </section>
    </div>
  );
}
