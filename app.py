import streamlit as st
from PIL import Image
from agents.vision_agent import VisionAgent
from agents.routine_agent import RoutineAgent
from agents.alert_agent import AlertAgent

st.set_page_config(page_title="Projet 15 — Téléassistance", layout="wide")
st.title("Projet 15 : Téléassistance & Détection de Chutes (3 Agents)")


@st.cache_resource
def load_vision():
    return VisionAgent("models/vision_cnn_best.keras")


vision = load_vision()
routine = RoutineAgent("models/rnn_routine.h5")
alert = AlertAgent("models/rf_triage.joblib")

immobile_sec = st.slider("Durée d'immobilité (secondes)", 0, 120, 20)
scenario = st.selectbox("Scénario de test", ["Chute + Inactivité", "Activité normale"])
uploaded = st.file_uploader("Image de la caméra", type=["png", "jpg", "jpeg"])

if st.button("Exécuter les 3 Agents"):
    if uploaded is None:
        st.warning("Veuillez charger une image pour l'Agent 1 (CNN).")
        st.stop()

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Image analysée", width=300)

    v_out = vision.analyze(image)
    r_out = routine.analyze(scenario)
    a_out = alert.decide(v_out, r_out, {"immobile_sec": immobile_sec})

    c1, c2, c3 = st.columns(3)
    c1.metric("Agent 1 (CNN Vision)", v_out["posture"], f"{v_out['fall_prob']:.0%}")
    c2.metric("Agent 2 (RNN Routine)", r_out["routine_state"], f"{r_out['anomaly_score']:.0%}")
    c3.metric("Agent 3 (RF Décision)", a_out["alert_level"], a_out["reason"])
