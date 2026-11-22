"""Streamlit demo for the CNN + RNN + alert fusion pipeline."""
from pathlib import Path
import streamlit as st
from PIL import Image
from agents.vision_agent import VisionAgent
from agents.routine_agent import RoutineAgent
from agents.alert_agent import AlertAgent

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
st.set_page_config(page_title="Projet 15 — Téléassistance", layout="wide")
st.title("Projet 15 : Téléassistance & Détection de Chutes")

@st.cache_resource
def load_vision():
    return VisionAgent(str(MODELS / "vision_cnn_best.keras"))

@st.cache_resource
def load_routine():
    return RoutineAgent(str(MODELS / "rnn_routine.h5"))

vision_path, routine_path = MODELS / "vision_cnn_best.keras", MODELS / "rnn_routine.h5"
rf_path = MODELS / "rf_triage.joblib"
missing = [str(p.relative_to(ROOT)) for p in (vision_path, routine_path) if not p.is_file()]
if missing:
    st.error("Required model file(s) missing: " + ", ".join(missing))
    st.stop()
try:
    vision, routine = load_vision(), load_routine()
except Exception as exc:
    st.error(f"Could not load CNN/RNN models: {exc}")
    st.stop()
alert = AlertAgent(rf_path)
if not rf_path.is_file():
    st.warning("Trained Random Forest not found. Alert decisions use the documented rule-based fallback; this is not an RF prediction.")

st.caption("Demo: the routine input is a generated scenario, not a live sensor stream.")
immobile_sec = st.slider("Durée d'immobilité (secondes)", 0, 120, 20)
scenario = st.selectbox("Scénario de routine", ["Chute + Inactivité", "Activité normale"])
uploaded = st.file_uploader("Image de la caméra", type=["png", "jpg", "jpeg"])
if st.button("Exécuter les 3 agents"):
    if uploaded is None:
        st.warning("Chargez une image pour lancer le CNN.")
        st.stop()
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Image analysée", width=300)
    vision_out = vision.analyze(image)
    routine_out = routine.analyze(scenario)
    alert_out = alert.decide(vision_out, routine_out, {"immobile_sec": immobile_sec})
    c1, c2, c3 = st.columns(3)
    c1.metric("CNN Vision", vision_out["posture"], f"{vision_out['fall_prob']:.0%}")
    c2.metric("RNN Routine", routine_out["routine_state"], f"{routine_out['anomaly_score']:.0%}")
    c3.metric("Alerte (" + alert_out["method"] + ")", alert_out["alert_level"], alert_out["reason"])
