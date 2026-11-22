"""Train from real paired predictions and human/ground-truth alert labels.

CSV must contain fall_prob, anomaly_score, alert_level. immobile_sec is optional.
No existing project dataset has these paired fusion labels, so training refuses
to substitute frame labels or generated scenarios.
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from preprocessing import FEATURES
from evaluation import LABELS, evaluate


def train(csv_path, model_path=None, results_path=None):
    df = pd.read_csv(csv_path)
    required = {"fall_prob", "anomaly_score", "alert_level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {', '.join(sorted(missing))}")
    if "immobile_sec" not in df:
        df["immobile_sec"] = 0.0
    if not set(df.alert_level.dropna().unique()).issubset(set(LABELS)):
        raise ValueError(f"alert_level values must be among {LABELS}")
    df = df.dropna(subset=["alert_level"])
    if df.alert_level.nunique() < 2:
        raise ValueError("At least two alert classes are required for training")
    X, y = df[list(FEATURES)], df.alert_level
    strat = y if y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=strat)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=42, n_jobs=-1))
    model.fit(Xtr, ytr)
    out = Path(model_path or Path(__file__).parent / "models/rf_triage.joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    metrics = evaluate(model, Xte, yte, results_path or Path(__file__).parent / "results/evaluation.json")
    print(f"Saved model: {out}\nHeld-out metrics: {metrics}")
    print("Feature importance:", dict(zip(FEATURES, model[-1].feature_importances_)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV of real paired CNN/RNN outputs and alert labels")
    parser.add_argument("--model")
    parser.add_argument("--results")
    args = parser.parse_args()
    train(args.csv, args.model, args.results)
