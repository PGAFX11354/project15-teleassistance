"""Train the RNN (GRU) of Agent 2 and save it to models/rnn_routine.h5.

Step 1 (once):  python -m training.prepare_har
Step 2:         python -m training.train_rnn
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from agents.routine_agent import day_signals, day_surprises, to_score
from agents.routine_sim import (A, ACTIVITIES, SLOTS_PER_DAY, awake_at_night,
                                inactive_afternoon, normal_day, observe)

MODEL_PATH = Path("models/rnn_routine.h5")
PREDICTIONS = Path("data/processed/har_predictions.csv")
N_ACT = len(ACTIVITIES)
NOISE = 0.10   # share of scrambled input activities during training


def load_pools():
    """Real classifier outputs, grouped by the real activity."""
    if not PREDICTIONS.exists():
        print(f"WARNING: {PREDICTIONS} not found. Run  python -m training.prepare_har  first.")
        print("Continuing with PERFECT activity labels (no real sensor data involved).\n")
        return None
    df = pd.read_csv(PREDICTIONS)
    return {A[a]: df.loc[df["true"] == a, "predicted"].map(A).to_numpy() for a in ACTIVITIES}


def build_model():
    acts = tf.keras.Input(shape=(None,), dtype="int32")    # activity at each slot
    slots = tf.keras.Input(shape=(None,), dtype="int32")   # time of day of the slot to predict
    x = tf.keras.layers.Concatenate()([
        tf.keras.layers.Embedding(N_ACT, 8)(acts),
        tf.keras.layers.Embedding(SLOTS_PER_DAY, 8)(slots),
    ])
    x = tf.keras.layers.GRU(32, return_sequences=True)(x)  # the RNN
    out = tf.keras.layers.Dense(N_ACT, activation="softmax")(x)
    model = tf.keras.Model([acts, slots], out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def main():
    rng = np.random.default_rng(0)
    tf.random.set_seed(0)
    pools = load_pools()

    def view(day):   # what the sensors + classifier would report for a true day
        return observe(day, pools, rng) if pools else day

    days = np.array([view(normal_day(rng)) for _ in range(800)])
    slots = np.tile(np.arange(1, SLOTS_PER_DAY), (len(days), 1))
    model = build_model()
    # Each epoch we scramble ~10% of the activities the model SEES (never the targets).
    # This stops it from just copying the last activity, and teaches it what is
    # expected at each time of day (e.g. nobody lies down at 2pm).
    for epoch in range(1, 26):
        seen = days[:, :-1].copy()
        scramble = rng.random(seen.shape) < NOISE
        seen[scramble] = rng.integers(0, N_ACT, scramble.sum())
        hist = model.fit([seen, slots], days[:, 1:], epochs=1, batch_size=32, verbose=0)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:2d}/25  loss {hist.history['loss'][0]:.3f}")

    # Reference level: what the worst normal days reach (97.5th percentile)
    normal_signals = np.array(
        [day_signals(day_surprises(model, view(normal_day(rng)))) for _ in range(300)])
    ref = np.percentile(normal_signals, 97.5, axis=0)

    def score(day):
        return to_score(day_signals(day_surprises(model, day)), ref)

    print("\nEvaluation on 100 fresh days of each type (flagged = score >= 0.5):")
    for name, make in [("normal day", normal_day),
                       ("inactive afternoon", inactive_afternoon),
                       ("awake at night", awake_at_night)]:
        scores = np.array([score(view(make(rng))) for _ in range(100)])
        print(f"  {name:20s} mean score {scores.mean():.2f} | flagged {np.mean(scores >= 0.5):.0%}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    info = {"ref": ref.tolist()}
    if pools:
        info["pools"] = {str(k): v.tolist() for k, v in pools.items()}
    MODEL_PATH.with_suffix(".json").write_text(json.dumps(info))
    print(f"\nSaved {MODEL_PATH} and {MODEL_PATH.with_suffix('.json')}")


if __name__ == "__main__":
    main()
