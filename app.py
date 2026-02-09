import streamlit as st
import numpy as np
import time
import random

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Smart Crowd Monitoring – College Gate",
    layout="wide"
)

st.title("🚦 Smart Crowd Monitoring System – College Gate")
st.caption("Real-time crowd density monitoring and congestion visualization")

# ==============================
# DEMO MODE FLAG
# ==============================
DEMO_MODE = True

# ==============================
# ZONE DEFINITIONS
# ==============================
ZONES = ["OUTSIDE", "GATE", "INSIDE"]

MAX_GATE_CAPACITY = 3

# ==============================
# SESSION STATE INIT
# ==============================
if "heatmap" not in st.session_state:
    st.session_state.heatmap = np.zeros((60, 100))

if "gate_history" not in st.session_state:
    st.session_state.gate_history = []

# ==============================
# LAYOUT
# ==============================
col1, col2 = st.columns([2, 1])

heatmap_placeholder = col1.empty()

outside_metric = col2.metric("Outside Count", 0)
gate_metric = col2.metric("Gate Count", 0)
inside_metric = col2.metric("Inside Count", 0)

risk_box = col2.empty()
action_box = col2.empty()

st.markdown("---")
st.markdown("### 🔍 System Status")

status = st.success("🟢 Live system running (Demo Mode)")

# ==============================
# MAIN LOOP
# ==============================
while True:
    # --------------------------
    # SIMULATED COUNTS (DEMO)
    # --------------------------
    outside = random.randint(0, 6)
    gate = random.randint(0, 4)
    inside = random.randint(0, 6)

    # Smooth gate count
    st.session_state.gate_history.append(gate)
    if len(st.session_state.gate_history) > 5:
        st.session_state.gate_history.pop(0)

    smooth_gate = int(sum(st.session_state.gate_history) / len(st.session_state.gate_history))

    # --------------------------
    # HEATMAP UPDATE
    # --------------------------
    heatmap = st.session_state.heatmap
    heatmap *= 0.90  # decay

    # Simulate crowd presence near gate
    for _ in range(smooth_gate):
        x = random.randint(40, 60)
        y = random.randint(20, 40)
        heatmap[y, x] += 3

    heatmap = np.clip(heatmap, 0, 255)
    st.session_state.heatmap = heatmap

    # --------------------------
    # RISK LOGIC
    # --------------------------
    density = smooth_gate / MAX_GATE_CAPACITY

    if density < 0.6:
        risk = "LOW"
        color = "🟢"
        action = "Normal flow"
    elif density < 0.85:
        risk = "MEDIUM"
        color = "🟡"
        action = "Monitor crowd. Prepare control."
    else:
        risk = "HIGH"
        color = "🔴"
        action = "STOP entry. Allow exit only!"

    # --------------------------
    # UPDATE DASHBOARD
    # --------------------------
    heatmap_placeholder.image(
        heatmap,
        clamp=True,
        channels="GRAY",
        caption="Crowd Density Heatmap (Red = High Congestion)"
    )

    outside_metric.metric("Outside Count", outside)
    gate_metric.metric("Gate Count", smooth_gate)
    inside_metric.metric("Inside Count", inside)

    risk_box.subheader(f"Risk Level: {color} {risk}")
    action_box.markdown(f"**Recommended Action:** {action}")

    time.sleep(1)
