"""Generates notebook/document_classification.ipynb from the cells below.

Keeping the notebook source in a .py file makes changes reviewable in git diffs.
Run:  python scripts/build_notebook.py
Then execute it (outputs saved in place):
      jupyter nbconvert --to notebook --execute --inplace notebook/document_classification.ipynb
"""
from pathlib import Path

import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Financial Document Classification – Training & Evaluation

**Use case.** This project builds on my Introduction to Machine Learning summative (*credit card default
risk prediction*, tabular data). Before a lender can score a credit application, the documents the
applicant submits (invoices, budgets, forms, letters, resumes, ...) must be sorted. This notebook trains an
**image** classifier that recognises the document type from a scanned page.

**Dataset.** [RVL-CDIP small-200](https://huggingface.co/datasets/vaclavpechtor/rvl_cdip-small-200), a
200-images-per-class subset of RVL-CDIP (Harley, Ufkes & Derpanis, *ICDAR 2015*). We keep the 9 document
types a lender would receive with a credit application.

**Contents.** 1) Data acquisition · 2) Exploration · 3) Preprocessing · 4) Model creation · 5) Training ·
6) Evaluation (accuracy, precision, recall, F1, confusion matrix, report) · 7) Prediction functions ·
8) Saving & registering the model.
""")

code("""
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))   # run from notebook/ -> import the project's src package
os.chdir(Path.cwd().parent)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from src.config import CLASS_NAMES, TRAIN_DIR, VAL_DIR, TEST_DIR, SAMPLES_DIR, MODELS_DIR, EPOCHS, BATCH_SIZE, IMG_SIZE
from src import preprocessing, model as M, registry
from src.dataset_stats import compute_stats

sns.set_theme(style="whitegrid")
print("TensorFlow", tf.__version__, "| classes:", CLASS_NAMES)
""")

md("""
## 1. Data acquisition
`scripts/download_data.py` downloads only our 9 classes from Hugging Face, converts the TIFF scans to
grayscale PNG and builds the splits:

* source *train* (160/class) → our **train (80%)** and **validation (20%)**, stratified with a fixed seed
* source *validation* (40/class) → our **test** set, held out and never used for training or model selection
""")

code("""
stats = compute_stats()
dist = stats["distribution"]
for split in ("train", "val", "test"):
    print(f"{split:5s}", sum(dist[split].values()), dist[split])
""")

md("## 2. Data exploration\n### 2.1 Example documents per class")

code("""
fig, axes = plt.subplots(2, 9, figsize=(20, 6))
for col, c in enumerate(CLASS_NAMES):
    files = sorted((TRAIN_DIR / c).glob("*.png"))[:2]
    for row, f in enumerate(files):
        axes[row, col].imshow(plt.imread(f), cmap="gray")
        axes[row, col].set_title(c if row == 0 else "", fontsize=11)
        axes[row, col].axis("off")
plt.tight_layout(); plt.show()
""")

md("### 2.2 Feature 1 – class distribution")

code("""
import pandas as pd
df_dist = pd.DataFrame(dist)
df_dist.plot(kind="bar", stacked=True, figsize=(10, 4), title="Images per document type and split")
plt.ylabel("images"); plt.show()
df_dist
""")

md("""
**Interpretation.** The dataset is perfectly balanced (128 train / 32 val / 40 test per type), so accuracy is
a fair headline metric and no class weighting or resampling is needed. In real lending data, invoices and
forms would dominate; this is a limitation discussed in the report.
""")

md("### 2.3 Feature 2 – page geometry (width, height, aspect ratio)")

code("""
sizes = pd.DataFrame(stats["sizes"])
feat = pd.DataFrame({c: {k: v["mean"] for k, v in f.items()} for c, f in stats["per_class_features"].items()}).T
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
sns.scatterplot(data=sizes, x="width", y="height", hue="class", ax=ax[0], s=25)
ax[0].set_title("Original page size (pixels)")
feat["aspect_ratio"].sort_values().plot(kind="barh", ax=ax[1], title="Mean aspect ratio (height / width)")
plt.tight_layout(); plt.show()
feat[["width", "height", "aspect_ratio"]].round(2)
""")

md("""
**Interpretation.** Almost every scan has the same height (~1000 px) with widths around 750–800 px – a portrait
page with aspect ratio ≈ 1.3. Page **shape carries almost no information about the class**, so the model has
to learn from the *layout* inside the page. Resizing to a square 224×224 slightly squashes pages, but it is
the same distortion for every class, so it does not bias the model.
""")

md("### 2.4 Feature 3 – ink density and brightness")

code("""
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
feat["ink_density"].sort_values().plot(kind="barh", ax=ax[0], color="#444", title="Mean ink density (share of dark pixels)")
feat["brightness"].sort_values().plot(kind="barh", ax=ax[1], color="#999", title="Mean brightness (0 = black, 255 = white)")
plt.tight_layout(); plt.show()
feat[["ink_density", "brightness"]].round(3)
""")

md("""
**Interpretation.** Document types differ clearly in how much ink they carry: dense, structured types
(forms, budgets, handwritten notes, questionnaires) have more dark pixels than sparse text types (letters,
memos). This visual difference is exactly what a CNN can learn. Types with similar ink density
(e.g. letter vs. memo vs. email) are where we expect the most confusion.
""")

md("""
## 3. Preprocessing
Implemented in `src/preprocessing.py` and inside the model (`src/model.py`):

| Step | Where | Why |
|---|---|---|
| Decode PNG/JPEG/TIFF with Pillow | `load_image` | one loader for every format users may upload |
| Convert to grayscale, then repeat to 3 channels | `load_image` | scans are black & white; MobileNetV2 expects RGB |
| Resize to 224×224 | `load_image` | MobileNetV2's native input size |
| Label encoding (name → index) | `encode_label` | fixed class order shared by training and the API |
| Augmentation: rotation ±2%, shift ±5%, zoom ±10% (training only) | model layer | imitates crooked scans → less overfitting with only 128 images/class |
| Normalise to [-1, 1] | model layer | MobileNetV2's expected range; inside the model so prediction can never use a different scaling |
""")

code("""
t0 = time.time()
x_train, y_train = preprocessing.load_dataset(TRAIN_DIR)
x_val, y_val = preprocessing.load_dataset(VAL_DIR)
x_test, y_test = preprocessing.load_dataset(TEST_DIR)
print(f"train {x_train.shape}, val {x_val.shape}, test {x_test.shape}  dtype={x_train.dtype}  ({time.time()-t0:.0f}s)")
""")

code("""
# Visualise preprocessing and augmentation on one invoice
sample = x_train[y_train == CLASS_NAMES.index("invoice")][0]
aug = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.02, fill_mode="constant", fill_value=255.0),
    tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=255.0),
    tf.keras.layers.RandomZoom(0.1, fill_mode="constant", fill_value=255.0)])
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
axes[0].imshow(sample); axes[0].set_title("preprocessed 224x224")
for i in range(1, 5):
    axes[i].imshow(aug(sample[None].astype("float32"), training=True)[0].numpy().astype("uint8"))
    axes[i].set_title(f"augmented #{i}")
for a in axes: a.axis("off")
plt.show()
""")

md("""
## 4. Model creation – MobileNetV2 transfer learning
**Why MobileNetV2?** With only 128 training images per class, training a CNN from scratch would overfit.
MobileNetV2 was pretrained on 1.2M ImageNet photos and already detects edges, lines, blocks and textures.
We reuse those features and train a new 9-class head. It is small (~3.5M parameters, ~10 MB) and fast on
CPU, which matters for serving many requests in the load test.
""")

code("""
model = M.build_model()
model.summary(show_trainable=True)
""")

md("""
## 5. Training
**Stage 1 – feature extraction:** backbone frozen, only the classifier head is trained (Adam, lr 1e-3).
**Stage 2 – fine-tuning:** the top 30 backbone layers are unfrozen with lr 1e-5 so they adapt from photos to
document layouts. Early stopping on validation accuracy (patience 4) restores the best weights.
""")

code("""
t0 = time.time()
history = M.train(model, x_train, y_train, x_val, y_val, epochs=EPOCHS, fine_tune_epochs=8, batch_size=BATCH_SIZE)
train_seconds = time.time() - t0
print(f"training time: {train_seconds/60:.1f} min")
""")

code("""
ft = history["fine_tune_start_epoch"]
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for metric, a in zip(("accuracy", "loss"), ax):
    a.plot(history[metric], label="train"); a.plot(history["val_" + metric], label="validation")
    a.axvline(ft - 0.5, ls="--", c="gray", label="fine-tuning starts")
    a.set_title(metric); a.set_xlabel("epoch"); a.legend()
plt.show()
""")

md("""
**Reading the curves.** Validation accuracy rises quickly in stage 1 (the pretrained features are already
useful) and improves again after fine-tuning starts. A gap where training accuracy is far above validation
accuracy would indicate overfitting; augmentation, dropout and early stopping are there to limit it.
""")

md("## 6. Evaluation on the held-out test set")

code("""
metrics = M.evaluate(model, x_test, y_test)
print(f"Accuracy          {metrics['accuracy']:.3f}")
print(f"Precision (macro) {metrics['precision_macro']:.3f}")
print(f"Recall (macro)    {metrics['recall_macro']:.3f}")
print(f"F1-score (macro)  {metrics['f1_macro']:.3f}")
""")

code("""
pd.DataFrame(metrics["classification_report"]).T.round(3)
""")

code("""
cm = np.array(metrics["confusion_matrix"])
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("predicted"); plt.ylabel("actual"); plt.title("Confusion matrix – test set")
plt.show()
""")

code("""
# Most frequent confusions (off-diagonal cells)
pairs = [(CLASS_NAMES[i], CLASS_NAMES[j], int(cm[i, j])) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j]]
pd.DataFrame(sorted(pairs, key=lambda p: -p[2])[:8], columns=["actual", "predicted", "count"])
""")

md("### Interpretation of the results\nSee the cell below – it is generated from the actual numbers.")

code("""
rep = metrics["classification_report"]
f1s = {c: rep[c]["f1-score"] for c in CLASS_NAMES}
best, worst = max(f1s, key=f1s.get), min(f1s, key=f1s.get)
print(f"Overall: {metrics['accuracy']:.1%} accuracy on {metrics['n_samples']} unseen documents "
      f"(random guessing = {1/len(CLASS_NAMES):.1%}).")
print(f"Easiest type: {best} (F1 {f1s[best]:.2f}); hardest type: {worst} (F1 {f1s[worst]:.2f}).")
print("Macro precision ≈ recall means errors are spread evenly rather than one class absorbing predictions.")
""")

md("""
**What this means for a lender.** Because the classes are balanced, macro-averaged precision, recall and
F1 weigh every document type equally. Precision answers *"when the model says invoice, how often is it
really an invoice?"*; recall answers *"of all invoices, how many did it find?"*. Confusions concentrate between
visually similar text-only types (letter / memo / email), while structured types with distinctive layouts are
recognised more reliably. In production, low-confidence predictions should be routed to a human reviewer
rather than processed automatically.
""")

md("## 7. Prediction functions")

code("""
from src.prediction import Predictor

def predict_document(path, model=model):
    \"\"\"Preprocess one image file and return (class, confidence, all probabilities).\"\"\"
    x = preprocessing.load_image(path)[None]
    probs = model(x, training=False).numpy()[0]
    return CLASS_NAMES[int(probs.argmax())], float(probs.max()), dict(zip(CLASS_NAMES, probs.round(3)))

samples = sorted(SAMPLES_DIR.glob("*.png"))
fig, axes = plt.subplots(2, 9, figsize=(22, 7))
for ax, f in zip(axes.flat, samples):
    label, conf, _ = predict_document(f)
    actual = f.name.split("_")[0]
    ax.imshow(plt.imread(f), cmap="gray"); ax.axis("off")
    ax.set_title(f"actual: {actual}\\npred: {label} ({conf:.0%})", color="green" if label == actual else "red", fontsize=10)
plt.tight_layout(); plt.show()
""")

md("## 8. Save and register the model (version 1)")

code("""
version = registry.next_version()
filename = f"doc_classifier_v{version}.h5"
M.save_model(model, MODELS_DIR / filename)
M.save_json(metrics, MODELS_DIR / "metrics.json")
M.save_json(history, MODELS_DIR / "history.json")
registry.record({
    "version": version, "file": filename, "kind": "initial", "status": "promoted",
    "created_at": registry.now_iso(), "train_samples": int(len(x_train)),
    "training_seconds": round(train_seconds, 1),
    "metrics": {k: metrics[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
}, promote=True)
print("saved", MODELS_DIR / filename, f"{(MODELS_DIR / filename).stat().st_size/1e6:.1f} MB")
""")

code("""
# Sanity check: the saved .h5 file reloads and the API's Predictor gives the same answer
p = Predictor()
f = samples[0]
print("notebook model:", predict_document(f)[:2])
print("reloaded model:", (lambda r: (r["predicted_class"], round(r["confidence"], 4)))(p.predict(f)), "version", p.version)
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent.parent / "notebook" / "document_classification.ipynb"
nbf.write(nb, out)
print("wrote", out)
