from integration import decide
from predict import MODEL_PATH


class AlertAgent:
    def __init__(self, model_path=MODEL_PATH, allow_rule_fallback=True):
        self.model_path = model_path
        self.allow_rule_fallback = allow_rule_fallback

    def decide(self, vision_out, routine_out, context=None):
        result = decide(
            vision_out=vision_out,
            routine_out=routine_out,
            context=context or {},
            model_path=self.model_path,
            allow_rule_fallback=self.allow_rule_fallback,
        )
        result["agent"] = "AlertAgent"
        return result
