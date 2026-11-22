"""Metrics and report serialization for a legitimately trained fusion model."""
from pathlib import Path
import json
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

LABELS = ["NORMAL", "WARNING", "EMERGENCY"]


def evaluate(model, X, y, output_path=None):
    pred = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=LABELS,
                                                  average="weighted", zero_division=0)
    result = {"accuracy": float(accuracy_score(y, pred)), "precision_weighted": float(p),
              "recall_weighted": float(r), "f1_weighted": float(f),
              "labels": LABELS, "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
              "classification_report": classification_report(y, pred, labels=LABELS,
                  output_dict=True, zero_division=0)}
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
