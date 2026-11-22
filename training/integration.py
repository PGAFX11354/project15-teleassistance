"""Independent adapter: existing CNN + RNN dictionaries -> final alert."""
from pathlib import Path
from training.preprocessing import prepare_features
from training.predict import MODEL_PATH, load_model, predict_level

_model = None


def decide(vision_out: dict, routine_out: dict, context: dict | None = None,
           model_path=MODEL_PATH, allow_rule_fallback=True) -> dict:
    """Return NORMAL/WARNING/EMERGENCY and provenance for teammate outputs.

    If a trained model is available it is used. Otherwise an explicit, documented
    threshold policy is used; it is not represented as a trained Random Forest.
    """
    global _model
    x = prepare_features(vision_out, routine_out, context)
    path = Path(model_path)
    if path.exists():
        if _model is None or getattr(_model, "_fusion_path", None) != str(path):
            _model = load_model(path)
            _model._fusion_path = str(path)
        level, method = predict_level(_model, x), "random_forest"
    elif allow_rule_fallback:
        fall, anomaly, immobile = x["fall_prob"], x["anomaly_score"], x["immobile_sec"]
        level = "EMERGENCY" if fall >= .80 and immobile >= 10 else (
            "WARNING" if fall >= .50 or anomaly >= .75 else "NORMAL")
        method = "rule_based_fallback"
    else:
        raise FileNotFoundError(f"No trained fusion model at {path}; train it or enable fallback")
    return {"agent": "RandomForestFusion", "alert_level": level, "method": method,
            "features": x, "reason": f"fall={x['fall_prob']:.2f}, anomaly={x['anomaly_score']:.2f}"}
