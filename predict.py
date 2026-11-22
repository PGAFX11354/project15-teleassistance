"""Inference helpers for the trained fusion classifier."""
from pathlib import Path
import joblib
from preprocessing import FEATURES, feature_matrix

MODEL_PATH = Path(__file__).parent / "models" / "rf_triage.joblib"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_level(model, features):
    return str(model.predict(feature_matrix([features]))[0])
