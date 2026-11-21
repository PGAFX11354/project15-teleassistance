import json
from pathlib import Path

import numpy as np

from agents.routine_sim import A, INACTIVE, SCENARIOS, normal_day, observe


def day_surprises(model, day):
    """How surprised the RNN is at each step of one day (-log of the probability
    it gave to what really happened). Low = expected, high = unusual."""
    day = np.asarray(day, dtype="int32")
    acts = day[None, :-1]
    slots = np.arange(1, len(day), dtype="int32")[None, :]
    probs = model.predict([acts, slots], verbose=0)[0]
    return -np.log(probs[np.arange(len(day) - 1), day[1:]] + 1e-9)


def day_signals(surprises):
    """Two simple signals describing how unusual a day is:
    (1) the biggest single surprise, (2) the total of all surprises above 1.
    A short strange event triggers (1), a long strange period triggers (2)."""
    return np.array([surprises.max(), surprises[surprises > 1].sum()])


def to_score(signals, ref):
    """Turn the signals into a 0-1 anomaly score. `ref` holds the level reached
    by the worst normal days: ratio 1.0 -> score 0.5, ratio 1.4 or more -> 1.0."""
    ratio = float(np.max(np.asarray(signals) / np.asarray(ref)))
    return float(np.clip((ratio - 0.6) / 0.8, 0.0, 1.0))


class RoutineAgent:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.ref = None
        self.pools = None
        self._rng = np.random.default_rng()
        if model_path and Path(model_path).exists():
            meta = Path(model_path).with_suffix(".json")
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path, compile=False)
                info = json.loads(meta.read_text())
                self.ref = np.array(info["ref"])
                if "pools" in info:   # real classifier outputs, used by the demo scenarios
                    self.pools = {int(k): np.array(v) for k, v in info["pools"].items()}
            except Exception as err:   # missing TensorFlow, missing json, ...
                print(f"[RoutineAgent] model not loaded ({err}); using stub output")
                self.model = None

    def _to_day(self, sequence_window):
        """Accepts a scenario name (demo app) or a list of activities, one per
        15-minute slot starting at midnight (names like 'SITTING' or ints 0-5)."""
        if isinstance(sequence_window, str):
            day = SCENARIOS.get(sequence_window, normal_day)(self._rng)
            return observe(day, self.pools, self._rng) if self.pools else day
        return np.array([A[x] if isinstance(x, str) else int(x) for x in sequence_window])

    def analyze(self, sequence_window):
        if self.model is None:   # no trained model: same dummy output as before
            return {"agent": "RoutineAgent", "routine_state": "Unusual Inactivity",
                    "anomaly_score": 0.84}

        day = self._to_day(sequence_window)
        s = day_surprises(self.model, day)
        score = to_score(day_signals(s), self.ref)

        if score < 0.5:
            state = "Normal Routine"
        else:
            odd = np.where(s > 1)[0]                 # the surprising moments
            if len(odd) == 0:
                odd = [int(np.argmax(s))]
            inactive_share = np.mean([day[1:][i] in INACTIVE for i in odd])
            state = "Unusual Inactivity" if inactive_share >= 0.5 else "Unusual Activity"

        return {"agent": "RoutineAgent", "routine_state": state,
                "anomaly_score": round(score, 2)}
