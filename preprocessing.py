"""Input normalization for the existing VisionAgent and RoutineAgent outputs."""
from __future__ import annotations

import numpy as np

FEATURES = ("fall_prob", "anomaly_score", "immobile_sec")


def _probability(value, name):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return value


def prepare_features(vision_out: dict, routine_out: dict, context: dict | None = None) -> dict:
    """Convert teammate output dictionaries into one named, validated feature row."""
    if "fall_prob" not in vision_out or "anomaly_score" not in routine_out:
        raise KeyError("Expected vision_out['fall_prob'] and routine_out['anomaly_score']")
    context = context or {}
    immobile = float(context.get("immobile_sec", 0.0))
    if not np.isfinite(immobile) or immobile < 0:
        raise ValueError("immobile_sec must be finite and non-negative")
    return {"fall_prob": _probability(vision_out["fall_prob"], "fall_prob"),
            "anomaly_score": _probability(routine_out["anomaly_score"], "anomaly_score"),
            "immobile_sec": immobile}


def feature_matrix(rows):
    """Create a consistently ordered numeric matrix from feature dictionaries."""
    import pandas as pd
    return pd.DataFrame(rows, columns=FEATURES)
