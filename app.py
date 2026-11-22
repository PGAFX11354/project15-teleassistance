import streamlit as st

from agents.vision_agent import VisionAgent
from agents.routine_agent import RoutineAgent
from agents.alert_agent import AlertAgent
from predict import MODEL_PATH

st.set_page_config(page_title="Projet 15 — Téléassistance", layout="wide")
st.title("Projet 15 : Téléassistance & Détection de Chutes (3 Agents)")
st.caption("Mode sans intrusion visuelle — simulation par scénarios et durée d'immobilité en mémoire.")


@st.cache_resource
def load_agents():
    vision = VisionAgent("models/vision_cnn_best.keras")
    routine = RoutineAgent("models/rnn_routine.h5")
    alert = AlertAgent(model_path=MODEL_PATH, allow_rule_fallback=True)
    return vision, routine, alert


vision, routine, alert = load_agents()

with st.sidebar:
    st.header("Paramètres de simulation")
    scenario = st.selectbox(
        "Scénario de test",
        ["Chute + Inactivité", "Activité normale"],
    )
    immobile_sec = st.slider("Durée d'immobilité (secondes)", 0, 120, 20)
    run_btn = st.button("Exécuter les 3 Agents", type="primary", use_container_width=True)

if run_btn:
    v_out = vision.analyze(scenario)
    r_out = routine.analyze(scenario)
    a_out = alert.decide(v_out, r_out, {"immobile_sec": immobile_sec})

    level = a_out.get("alert_level", "NORMAL")
    method = a_out.get("method", "rule_based_fallback")

    if level == "EMERGENCY":
        st.error(f"ALERTE CRITIQUE : {level}  |  Méthode : `{method}`")
    elif level == "WARNING":
        st.warning(f"VIGILANCE : {level}  |  Méthode : `{method}`")
    else:
        st.success(f"ÉTAT NORMAL : {level}  |  Méthode : `{method}`")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("1. VisionAgent (CNN)")
        st.metric("Posture", v_out.get("posture", "N/A"))
        st.metric("Probabilité de chute", f"{v_out.get('fall_prob', 0.0):.2%}")

    with c2:
        st.subheader("2. RoutineAgent (RNN)")
        st.metric("État de routine", r_out.get("routine_state", "N/A"))
        st.metric("Score d'anomalie", f"{r_out.get('anomaly_score', 0.0):.2%}")

    with c3:
        st.subheader("3. AlertAgent (Fusion)")
        st.metric("Niveau d'alerte", level)
        st.write(f"**Immobilité :** {immobile_sec} s")
        st.write(f"**Méthode :** `{method}`")
        if a_out.get("reason"):
            st.write(f"**Motif :** {a_out['reason']}")

    with st.expander("Détails bruts des sorties agents"):
        st.json({"vision": v_out, "routine": r_out, "alert": a_out})
