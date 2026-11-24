"""Create rnn_scores.csv for the Random Forest: one RNN anomaly score per sequence.

Usage (project root):
  python -m training.make_rnn_scores                  -> 70 sequences, no input file needed
  python -m training.make_rnn_scores cnn_scores.csv   -> one row per seq_id of that file
                                                         (needs columns: seq_id, label)
Output: data/processed/rnn_scores.csv  (seq_id, anomaly_score, routine_state, paired_day)

IMPORTANT: no dataset has camera sequences AND daily routines of the same person.
So each camera sequence is PAIRED with a simulated day, and that pairing is an
ASSUMPTION (the two numbers below), not a measurement. State it in the report,
and do NOT use the column `paired_day` as a feature of the Random Forest.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agents.routine_agent import RoutineAgent

P_UNUSUAL_IF_FALL = 0.6   # assumption: after a fall, the day is often unusual
P_UNUSUAL_IF_ADL = 0.1    # assumption: during daily activities, rarely unusual
P_UNUSUAL_NO_FILE = 0.3   # share of unusual days when no input file is given
N_NO_FILE = 70            # number of sequences when no input file is given
OUT = Path("data/processed/rnn_scores.csv")


def main():
    if len(sys.argv) > 2:
        sys.exit("Usage: python -m training.make_rnn_scores [path/to/cnn_scores.csv]")
    if len(sys.argv) == 2:
        cnn = pd.read_csv(sys.argv[1])
        missing = {"seq_id", "label"} - set(cnn.columns)
        if missing:
            sys.exit(f"Missing columns in {sys.argv[1]}: {sorted(missing)}")
        cases = list(zip(cnn["seq_id"], cnn["label"].map({1: P_UNUSUAL_IF_FALL, 0: P_UNUSUAL_IF_ADL})))
    else:
        cases = [(i, P_UNUSUAL_NO_FILE) for i in range(1, N_NO_FILE + 1)]

    agent = RoutineAgent("models/rnn_routine.h5")
    if agent.model is None:
        sys.exit("RNN model not found. Run  python -m training.train_rnn  first.")

    rng = np.random.default_rng(0)
    rows = []
    for seq_id, p_unusual in cases:
        unusual = rng.random() < p_unusual
        scenario = "Chute + Inactivité" if unusual else "Activité normale"
        out = agent.analyze(scenario)
        rows.append({"seq_id": seq_id, "anomaly_score": out["anomaly_score"],
                     "routine_state": out["routine_state"],
                     "paired_day": "unusual" if unusual else "normal"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.groupby("paired_day")["anomaly_score"].describe()[["count", "mean", "min", "max"]])
    print(f"\nSaved {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
