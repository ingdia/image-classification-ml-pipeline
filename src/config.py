"""Central configuration: paths, classes and hyper-parameters.

Every other module imports from here so that training, prediction, retraining
and the API always agree on image size, class order and file locations.
Paths can be overridden with environment variables (see .env.example).
"""
import os
from pathlib import Path

ROOT_DIR = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))
CANDIDATES_DIR = MODELS_DIR / "candidates"
REGISTRY_PATH = MODELS_DIR / "registry.json"

RUNTIME_DIR = Path(os.getenv("RUNTIME_DIR", ROOT_DIR / "runtime"))
RETRAIN_STATUS_PATH = RUNTIME_DIR / "retrain_status.json"
RETRAIN_LOCK_PATH = RUNTIME_DIR / "retrain.lock"

# The 9 RVL-CDIP document types a lender receives with a credit application.
# Order matters: index i in the model output corresponds to CLASS_NAMES[i].
CLASS_NAMES = [
    "budget",
    "email",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "questionnaire",
    "resume",
]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = (224, 224)  # MobileNetV2 input resolution
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EPOCHS = int(os.getenv("EPOCHS", 15))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", 5))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-3))
SEED = 42

# A retrained model is promoted only if its test accuracy is at least
# (current accuracy - PROMOTION_TOLERANCE).
PROMOTION_TOLERANCE = float(os.getenv("PROMOTION_TOLERANCE", 0.01))

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 20))
