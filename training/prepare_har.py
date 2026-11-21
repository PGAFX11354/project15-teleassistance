"""Step 1: turn the REAL sensor data into 'what an activity classifier would say'.

A Random Forest learns to recognise the activity from the 561 sensor features.
Each row is predicted by a model that never saw that volunteer (cross-validation
by subject), so the mistakes are realistic. The result is saved in
data/processed/har_predictions.csv and used by train_rnn.py.

Usage (project root):  python -m training.prepare_har
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict

CSV = Path("data/raw/train.csv")
OUT = Path("data/processed/har_predictions.csv")


def main():
    df = pd.read_csv(CSV)
    X = df.drop(columns=["subject", "Activity"])
    y = df["Activity"]

    print(f"{len(df)} real sensor windows from {df['subject'].nunique()} volunteers")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    pred = cross_val_predict(rf, X, y, groups=df["subject"], cv=GroupKFold(n_splits=5))

    print(f"Accuracy on volunteers the model has never seen: {accuracy_score(y, pred):.1%}")
    print("\nWhat the classifier says (rows = real activity, columns = predicted):")
    print(pd.crosstab(y, pred).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"true": y.values, "predicted": pred}).to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
