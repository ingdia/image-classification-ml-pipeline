import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, pct } from "../api.js";

// Single document prediction: upload (or pick a sample) -> type + confidence for all 9 types.
export default function PredictTab() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [drag, setDrag] = useState(false);
  const [samples, setSamples] = useState([]);
  const [expected, setExpected] = useState(null);

  useEffect(() => { api.samples().then((s) => setSamples(s.samples)).catch(() => {}); }, []);
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  const choose = (f, label = null) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setExpected(label);
  };

  const pickSample = async (s) => {
    const blob = await (await fetch(api.sampleUrl(s.name))).blob();
    choose(new File([blob], s.name, { type: "image/png" }), s.label);
  };

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      setResult(await api.predict(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const chartData = result
    ? Object.entries(result.probabilities).map(([name, p]) => ({ name, p: +(p * 100).toFixed(1) })).sort((a, b) => b.p - a.p)
    : [];

  return (
    <div className="grid">
      <section className="card stack">
        <h2>1. Choose a scanned document</h2>
        <label
          className={`dropzone ${drag ? "drag" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); choose(e.dataTransfer.files[0]); }}
        >
          <input type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp" hidden onChange={(e) => choose(e.target.files[0])} />
          Drop an image here or <u>browse</u> (PNG, JPG, TIFF, BMP)
        </label>
        {samples.length > 0 && (
          <div>
            <h3>…or try a test document the model has never seen</h3>
            <div className="samples">
              {samples.map((s) => <button key={s.name} onClick={() => pickSample(s)}>{s.label}</button>)}
            </div>
          </div>
        )}
        {preview && <img className="preview" src={preview} alt="selected document" />}
        <div className="row">
          <button className="primary" disabled={!file || loading} onClick={run}>
            {loading ? "Predicting…" : "Predict document type"}
          </button>
          {file && <span className="muted">{file.name}</span>}
        </div>
        {error && <div className="error">{error}</div>}
      </section>

      <section className="card">
        <h2>2. Prediction</h2>
        {!result && <p className="muted">The predicted document type and the model's confidence for every type will appear here.</p>}
        {result && (
          <div className="stack">
            <div>
              <div className="muted">Predicted type</div>
              <div className="result-main">{result.predicted_class}</div>
              <div>Confidence <strong>{pct(result.confidence)}</strong></div>
              {expected && (
                <div className={expected === result.predicted_class ? "pill ok" : "pill bad"} style={{ display: "inline-block", marginTop: 6 }}>
                  True type: {expected} — {expected === result.predicted_class ? "correct" : "incorrect"}
                </div>
              )}
              {result.confidence < 0.5 && (
                <div className="notice" style={{ marginTop: 8 }}>Low confidence: in a real lending workflow this document would be sent to a human reviewer.</div>
              )}
            </div>
            <div style={{ width: "100%", height: 300 }}>
              <ResponsiveContainer>
                <BarChart data={chartData} layout="vertical" margin={{ left: 30, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" domain={[0, 100]} unit="%" />
                  <YAxis type="category" dataKey="name" width={95} />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="p" name="probability">
                    {chartData.map((d) => <Cell key={d.name} fill={d.name === result.predicted_class ? "#1f5fbf" : "#a9bcd8"} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="muted">Model v{result.model_version} · inference {result.inference_ms} ms · container {result.container?.slice(0, 6)}</p>
          </div>
        )}
      </section>
    </div>
  );
}
