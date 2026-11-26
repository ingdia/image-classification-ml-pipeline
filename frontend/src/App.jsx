import { useState } from "react";
import StatusBar from "./components/StatusBar.jsx";
import PredictTab from "./components/PredictTab.jsx";
import DataTab from "./components/DataTab.jsx";
import RetrainTab from "./components/RetrainTab.jsx";
import ModelTab from "./components/ModelTab.jsx";

const TABS = [
  { id: "predict", label: "Predict", component: PredictTab },
  { id: "data", label: "Data & Insights", component: DataTab },
  { id: "retrain", label: "Upload & Retrain", component: RetrainTab },
  { id: "model", label: "Model Performance", component: ModelTab },
];

export default function App() {
  const [tab, setTab] = useState("predict");
  const Active = TABS.find((t) => t.id === tab).component;
  return (
    <>
      <header className="header">
        <div className="header-top">
          <div>
            <h1>Financial Document Classifier</h1>
            <p className="subtitle">Sorts the scanned documents a lender receives with a credit application</p>
          </div>
          <StatusBar />
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        <Active />
      </main>
    </>
  );
}
