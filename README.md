# Projet 15 — Téléassistance & Détection de Chutes

Streamlit demo combining the supplied CNN fall probability, RNN routine anomaly score, and a final alert decision. The CNN and RNN are independent agents under `agents/`; `AlertAgent` sends their output dictionaries to the Random Forest fusion adapter in `training/`.

## Structure

```text
agents/vision_agent.py       CNN image inference
agents/routine_agent.py      RNN routine anomaly inference
agents/routine_sim.py        Generated routine scenarios used by the demo
agents/alert_agent.py        RF/rule-fallback integration adapter
models/vision_cnn_best.keras CNN model
models/rnn_routine.h5        RNN model
models/rnn_routine.json      RNN reference thresholds and activity pools
models/rf_triage.joblib      Optional trained fusion model (not currently available)
training/train.py           Preserved training script
training/train_rf.py        RF training entry point
training/preprocessing.py   Input feature validation
training/evaluation.py      Evaluation metrics
training/predict.py         Model loading and prediction
training/integration.py     Inference and agent adapter
app.py                      Streamlit application
```

## Install and run

Python 3.12 is recommended. TensorFlow support depends on platform; install the pinned-compatible wheel available for your OS/Python version.

```shell
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The CNN and RNN artifacts in `models/` were copied from the existing teammate work. Their sizes are approximately 2.98 MB and 105 KB, well below GitHub's 100 MB per-file limit. The RNN JSON is also required for its score calibration. The demo accepts an image for the CNN and generated scenarios for the RNN; this is not a live sensor integration.

## Random Forest training

**There is currently no `rf_triage.joblib`.** The datasets and files inspected contain no paired CNN/RNN predictions with justified NORMAL/WARNING/EMERGENCY ground-truth alert labels. UR Fall frame labels and HAR activity labels cannot serve as those fusion labels. No RF has been trained and no RF evaluation metrics are claimed.

When real labeled paired records are available, create a CSV with columns `fall_prob,anomaly_score,alert_level`; `immobile_sec` is optional and defaults to 0. Each row must pair outputs from the actual CNN and RNN with a defensible alert outcome. Then run:

```shell
python -m training.train_rf path/to/paired_labeled_predictions.csv
```

This writes `models/rf_triage.joblib` and `results/evaluation.json` and prints held-out accuracy, weighted precision/recall/F1, confusion matrix, and feature importances. Do not interpret metrics from a tiny or unrepresentative dataset as reliable.

Until that artifact is produced, the alert adapter uses a clearly disclosed rule-based fallback (emergency when fall probability >= 0.80 and immobility >= 10 seconds; warning when fall probability >= 0.50 or anomaly >= 0.75; otherwise normal). The Streamlit UI labels the method. It is not a trained Random Forest.

## Known limitations

* No trained RF artifact or alert-level fusion dataset is present.
* Routine scenario inputs are generated simulations; the checked-in RNN falls back to a fixed stub if its model cannot load.
* The current demo processes a user-provided image. Verify that the CNN is trained for the intended silhouette/pose input and privacy requirement before deployment.
* Alert thresholds are demo policy only, not validated clinical or emergency dispatch criteria.
