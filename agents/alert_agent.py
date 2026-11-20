class AlertAgent:
    def __init__(self, model_path=None):
        self.model_path = model_path

    def decide(self, vision_out, routine_out, context):
        if vision_out["fall_prob"] > 0.8 and context.get("immobile_sec", 0) > 10:
            level = "ALERT_SAMU"
        elif vision_out["fall_prob"] > 0.5 or routine_out["anomaly_score"] > 0.75:
            level = "NOTIFY_FAMILY"
        else:
            level = "SAFE"

        return {
            "agent": "AlertAgent",
            "alert_level": level,
            "reason": f"fall={vision_out['fall_prob']:.2f}, anomaly={routine_out['anomaly_score']:.2f}",
        }
