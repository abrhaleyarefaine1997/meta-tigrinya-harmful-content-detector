from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TRAIN_PATH = DATA_DIR / "tig_train.csv"
TEST_PATH = DATA_DIR / "tig_test.csv"

MODEL_PATH = MODEL_DIR / "xgb_model.json"