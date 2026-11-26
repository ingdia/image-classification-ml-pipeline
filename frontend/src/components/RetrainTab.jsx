import { useCallback, useEffect, useRef, useState } from "react";
import { api, pct } from "../api.js";

const CLASSES = ["budget", "email", "form", "handwritten", "invoice", "letter", "memo", "questionnaire", "resume"];
const DONE = ["promoted", "rejected", "failed", "idle"];

// Bulk upload of labelled data + retraining trigger + live training status.
export default function RetrainTab() {
  const [label, setLabel] = useState("invoice");
  const [files, setFiles] = useState([]);
  const [uploadResult, setUploadResult] = useState(null);
  const [pending, setPending] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef();

  const refresh = useCallback(async () => {
    try {
      const [p, s] = await Promise.all([api.uploads(), api.retrainStatus()]);
      setPending(p);
      setStatus(s);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll status every 2 s while a retraining job is running.
  useEffect(() => {
    if (!status || DONE.includes(status.state)) return;
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [status, refresh]);

  const doUpload = async () => {
    setBusy(true); setError(null); setUploadResult(null);
    try {
      setUploadResult(await api.upload(files, label));
      setFiles([]);
      if (inputRef.current) inputRef.current.value = "";
    } catch (e) {
      setError(e.message);
      if (e.body?.detail?.files) setUploadResult(e.body.detail);
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const doRetrain = async () => {
    setError(null);
    try { await api.retrain(); } catch (e) { setError(e.message); }
    refresh();
  };

  const doClear = async () => { await api.clearUploads().catch((e) => setError(e.message)); refresh(); };

  const running = status && !DONE.includes(status.state);
  const prog = status?.progress;
  const res = status?.result;

  return (
    <div className="grid">
      <section className="card stack">
        <h2>1. Upload new labelled documents</h2>
        <p className="muted">
          Select many images (or a ZIP). For a ZIP with one folder per type (e.g. <code>invoice/a.png</code>) the folder
          name is used as the label; otherwise the type chosen below is used. Every file is validated: image type,
          size, whether it can be decoded, and minimum resolution.
        </p>
        <div className="row">
          <label>Document type&nbsp;
            <select value={label} onChange={(e) => setLabel(e.target.value)}>
              {CLASSES.map((c) => <option key={c}>{c}</option>)}
            </select>
          </label>
          <input ref={inputRef} type="file" multiple accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.zip"
                 onChange={(e) => setFiles([...e.target.files])} />
        </div>
        <div className="row">
          <button className="primary" disabled={!files.length || busy || running} onClick={doUpload}>
            {busy ? "Uploading…" : `Upload ${files.length || ""} file${files.length === 1 ? "" : "s"}`}
          </button>
        </div>
        {uploadResult && (
          <div>
            <div className={uploadResult.valid ? "notice" : "error"}>
              {uploadResult.valid} accepted, {uploadResult.invalid} rejected
            </div>
            {uploadResult.invalid > 0 && (
              <div className="table-wrap" style={{ marginTop: 8 }}>
                <table>
                  <thead><tr><th>File</th><th>Reason rejected</th></tr></thead>
                  <tbody>
                    {uploadResult.files.filter((f) => !f.valid).map((f, i) => (
                      <tr key={i}><td>{f.filename}</td><td>{f.reason}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
        <div>
          <h3>Waiting for the next retraining: {pending?.total ?? 0} images</h3>
          <div className="samples">
            {pending && Object.entries(pending.per_class).filter(([, n]) => n > 0).map(([c, n]) => (
              <span key={c} className="pill">{c}: {n}</span>
            ))}
          </div>
          {pending?.total > 0 && !running && (
            <button className="secondary" style={{ marginTop: 8 }} onClick={doClear}>Clear pending uploads</button>
          )}
        </div>
      </section>

      <section className="card stack">
        <h2>2. Retrain the model</h2>
        <p className="muted">
          Retraining fine-tunes a <strong>copy</strong> of the live model on the original training data plus your uploads,
          then tests it on the same held-out test set. It is promoted only if its accuracy is not more than 1 percentage
          point below the live model; otherwise it is rejected and the live model keeps serving.
        </p>
        <button className="primary" disabled={running || !pending?.total} onClick={doRetrain}>
          {running ? "Retraining in progress…" : "Retrain model"}
        </button>
        {!pending?.total && !running && <span className="muted">Upload data first to enable retraining.</span>}
        {error && <div className="error">{error}</div>}

        <div>
          <h3>Training / retraining status</h3>
          {status && (
            <div className="stack">
              <div className="row">
                <span className={`badge ${status.state}`}>{status.state}</span>
                <span className="muted">{status.message}</span>
              </div>
              {running && prog && (
                <div>
                  <div className="progress"><div style={{ width: `${(100 * prog.epoch) / prog.total_epochs}%` }} /></div>
                  <div className="muted">Epoch {prog.epoch} of {prog.total_epochs}</div>
                </div>
              )}
              {res && !running && (
                <table>
                  <tbody>
                    <tr><th>Candidate</th><td>v{res.version} ({res.status})</td></tr>
                    <tr><th>Candidate accuracy</th><td>{pct(res.metrics?.accuracy)}</td></tr>
                    <tr><th>Live model compared to</th><td>v{res.compared_to?.version}: {pct(res.compared_to?.accuracy)}</td></tr>
                    <tr><th>Training images</th><td>{res.train_samples} ({res.new_samples} new)</td></tr>
                    <tr><th>Duration</th><td>{res.training_seconds} s</td></tr>
                  </tbody>
                </table>
              )}
              {status.finished_at && <div className="muted">Finished {new Date(status.finished_at).toLocaleString()}</div>}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
