# Financial Document Classification – End-to-End ML Pipeline

An end-to-end machine learning pipeline that classifies the **scanned documents a lender receives with a
credit application** into 9 types: budget, email, form, handwritten, invoice, letter, memo, questionnaire and
resume. It covers data acquisition, preprocessing, transfer-learning model training, evaluation, a FastAPI
backend, a React dashboard, bulk upload and retraining with a promotion gate, Docker scaling behind an Nginx
load balancer, and Locust load testing.

African Leadership University – BSE – *Machine Learning Pipeline* summative.

| | |
|---|---|
| Video demo | _YouTube link: to be added_ |
| Live app | _Deployment URL: to be added_ |
| Notebook | [`notebook/document_classification.ipynb`](notebook/document_classification.ipynb) |
| Model file | [`models/doc_classifier_v1.h5`](models/) (Keras HDF5) |

---

## 1. Problem statement

This project builds on my *Introduction to Machine Learning* summative, **credit card default risk
prediction**, which scored clients from tabular data. Before any credit decision, a lender has to sort the
documents an applicant submits: invoices and budgets as evidence of income and spending, resumes and letters
as employment evidence, and forms and questionnaires from the application itself. Doing this by hand is slow
and error-prone.

**Goal:** given a scanned page, predict its document type so it can be routed automatically, with
low-confidence pages sent to a human reviewer. The data is **images** rather than tables, as the brief
requires.

## 2. Dataset

* **Source:** [RVL-CDIP small-200](https://huggingface.co/datasets/vaclavpechtor/rvl_cdip-small-200) on
  Hugging Face. It is a 200-image-per-class subset of **RVL-CDIP** (Harley, Ufkes & Derpanis, *"Evaluation
  of Deep Convolutional Nets for Document Image Classification and Retrieval"*, ICDAR 2015;
  [homepage](https://www.cs.cmu.edu/~aharley/rvl-cdip/)).
* **Classes used:** the 9 of 16 types relevant to lending. Advertisements, scientific papers, news articles
  and similar types are excluded.
* **Splits:** 128 train / 32 validation / 40 test images per class, giving **1,152 / 288 / 360 images**
  (1,800 in total). The test set is the source's validation split and is never used for training or model
  selection.
* The full dataset is **not committed** because of its size. `scripts/download_data.py` downloads it and
  rebuilds `data/train`, `data/val` and `data/test`. Eighteen sample test images (2 per type) are committed
  in `data/samples/`.

## 3. Architecture

```
                ┌──────────────────────── Docker Compose ────────────────────────┐
 Browser ──────►│ nginx :8080                                                     │
 Locust  ──────►│  ├─ /            → React dashboard (static build)               │
                │  └─ /api/*  ──round-robin──► api_1 │ api_2 │ api_3 (FastAPI+TF) │
                │                                   │                             │
                │              shared volumes: models/ (model + registry.json)    │
                │                              data/   (dataset + uploads)        │
                │                              runtime/ (retraining status/lock)  │
                └─────────────────────────────────────────────────────────────────┘
```

| Layer | Technology | Files |
|---|---|---|
| Data acquisition | Hugging Face Hub, Pillow | `scripts/download_data.py` |
| Preprocessing | Pillow, NumPy | `src/preprocessing.py` |
| Model | TensorFlow/Keras **MobileNetV2** (transfer learning) | `src/model.py`, `src/train.py` |
| Prediction | Keras, hot-reloading predictor | `src/prediction.py` |
| Retraining | background thread + promotion gate + model registry | `src/retrain.py`, `src/registry.py`, `src/upload_validation.py` |
| API | FastAPI + Uvicorn (Swagger at `/docs`) | `src/api/main.py` |
| Dashboard | React (Vite) + Recharts | `frontend/` |
| Load balancing | Nginx (Docker DNS round-robin) | `docker/nginx.conf` |
| Containers | Docker, Docker Compose | `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` |
| Load testing | Locust | `locust/`, `scripts/run_load_tests.sh` |

## 4. Repository structure

```
README.md  Dockerfile  docker-compose.yml  requirements.txt  requirements-dev.txt  .env.example
notebook/document_classification.ipynb   training + evaluation notebook (executed, with outputs)
src/        preprocessing.py  model.py  prediction.py  train.py  retrain.py  registry.py
            upload_validation.py  dataset_stats.py  config.py  api/main.py
data/       train/ val/ test/ (downloaded)   samples/ (committed)   dataset_stats.json   uploads/
models/     doc_classifier_v1.h5  registry.json  metrics.json  history.json
frontend/   React dashboard + Dockerfile
docker/     nginx.conf
locust/     locustfile.py  plot_results.py  results/
tests/      pytest suite (preprocessing, prediction, registry, upload validation, retraining, API)
scripts/    download_data.py  build_notebook.py  run_load_tests.sh
```

## 5. Setup and running

**Requirements:** Docker with Docker Compose v2. Node 20+ is only needed for frontend development.

```bash
git clone https://github.com/ingdia/image-classification-ml-pipeline.git
cd image-classification-ml-pipeline
cp .env.example .env                       # optional: adjust settings

# 1. Download the dataset (about 1 minute; builds data/train, data/val, data/test)
docker compose --profile dev run --rm dev python scripts/download_data.py

# 2. Start the app: API + nginx + dashboard
docker compose up -d --build
#    Dashboard:    http://localhost:8080
#    Swagger docs: http://localhost:8080/docs

# 3. Scale to 3 API containers behind the load balancer
docker compose up -d --scale api=3
```

The trained model `models/doc_classifier_v1.h5` is committed, so the app works without retraining.
Prediction does not need the dataset. Retraining and live evaluation do need it.

### Train from scratch

```bash
docker compose --profile dev run --rm dev python -m src.train          # command-line training
# or run the notebook and save its outputs:
docker compose --profile dev run --rm dev jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 notebook/document_classification.ipynb
```

### Run the tests

```bash
docker compose --profile dev run --rm dev pytest -q
```

## 6. Using the pipeline

### Predict
Open the dashboard's **Predict** tab. Drop a scanned page, or click a sample type to load a test document,
then click **Predict**. The predicted type, its confidence, and the probability for every type are shown.

### Upload data and retrain
1. **Upload & Retrain** tab: choose a document type, select many images (or a ZIP with one folder per type),
   then click **Upload**. Each file is validated: allowed type, size limit, decodable image, minimum
   resolution and known label. Rejected files are listed with the reason.
2. Click **Retrain model**. A copy of the live model is fine-tuned in the background on the original
   training data plus the uploads. Progress is shown per epoch.
3. **Promotion gate:** the candidate is evaluated on the fixed test set and promoted only if
   `candidate accuracy ≥ live accuracy − 1 pp`. Otherwise it is **rejected** and the live model keeps
   serving. Every version is recorded in `models/registry.json`. All API containers hot-reload a promoted
   model without restarting.

### Monitor and evaluate in production
The header shows **online status, uptime, model version and the container that answered**, refreshed every
5 seconds. The **Model Performance** tab shows test metrics, the confusion matrix, training curves and the
version history. **Evaluate live model now** re-scores the currently deployed model on the test set inside the
running container.

## 7. API

Interactive documentation is at `/docs` (Swagger UI). Through nginx, every path is prefixed with `/api`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, uptime, model version, container id |
| POST | `/predict` | Predict the type of one image (`file`) |
| POST | `/upload` | Bulk upload of labelled images or ZIPs (`files`, `label`) for retraining |
| POST | `/validate` | Dry-run validation of files without saving them |
| GET / DELETE | `/uploads` | Uploads waiting for the next retraining, or clear them |
| POST | `/retrain` | Start background retraining (202, or 409 if one is already running) |
| GET | `/retrain/status` | queued / running (epoch progress) / promoted / rejected / failed |
| GET | `/metrics` | Live model test metrics, training history, version history |
| POST | `/evaluate` | Evaluate the live model on the test set now (evaluation in production) |
| GET | `/dataset/stats` | Dataset features used by the visualisations |
| GET | `/samples`, `/samples/{name}` | Demo test documents |

Example:
```bash
curl -F "file=@data/samples/invoice_0000327323.png" http://localhost:8080/api/predict
```

## 8. Model results

_Filled in from the final executed notebook. See section 6 of the notebook and `models/metrics.json`._

## 9. Load testing (flood request simulation)

_Filled in with measured Locust results._

## 10. Deployment

_Filled in after deployment._

## 11. Limitations

* With 1,152 training images and pages downscaled to 224×224, small text is unreadable, so the model relies
  on layout. Visually similar types (form vs. questionnaire, letter vs. memo) are often confused.
* The documents are 1980s–90s US corporate scans (RVL-CDIP), not African bank documents.
* Retraining runs in a thread inside the API container. A production system would use a job queue
  (for example Celery + Redis) and a model registry service (for example MLflow).
