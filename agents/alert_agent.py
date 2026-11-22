"""Alert adapter for the CNN and RNN agents."""
from pathlib import Path
from training.integration import decide

class AlertAgent:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path) if model_path else Path(__file__).resolve().parents[1] / "models" / "rf_triage.joblib"

    def decide(self, vision_out, routine_out, context=None):
        return decide(vision_out, routine_out, context, model_path=self.model_path)
