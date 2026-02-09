import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==============================
# STREAMLIT PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Smart Crowd Monitoring – College Gate",
    layout="wide"
)

st.title("🚦 Smart Crowd Monitoring System – College Gate")
st.caption("Real-time crowd density, risk alerts, and heatmap visualization")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ==============================
# VIDEO SOURCE
# ==============================
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("videos/gate_demo.mp4")

# ==============================
# FRAME SIZE
# ==============================
FRAME_WIDTH = 800
FRAME_HEIGHT = 480

# ==============================
# ZONES
# ==============================
ZONE_OUTSIDE = (0, 0, 300, FRAME_HEIGHT)
ZONE_GATE    = (300, 0, 450, FRAME_HEIGHT)
ZONE_INSIDE  = (450, 0, FRAME_WIDTH, FRAME_HEIGHT)

ZONES = {
    "OUTSIDE": ZONE_OUTSIDE,
    "GATE": ZONE_GATE,
    "INSIDE": ZONE_INSIDE
}

def in_zone(cx, cy, zone):
    x1, y1, x2, y2 = zone
    return x1 < cx < x2 and y1 < cy < y2

# ==============================
# PARAMETERS
# ==============================
MAX_GATE_CAPACITY = 3
gate_history = deque(maxlen=10)

heatmap = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)

# ==============================
# STREAMLIT PLACEHOLDERS
# ==============================
col1, col2 = st.columns([2, 1])

frame_placeholder = col1.empty()

outside_metric = col2.metric("Outside Count", 0)
gate_metric = col2.metric("Gate Count", 0)
inside_metric = col2.metric("Inside Count", 0)

risk_box = col2.empty()
action_box = col2.empty()

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera feed not available")
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Heatmap decay
    heatmap *= 0.95

    # YOLO detection
    results = model(frame, classes=[0], conf=0.25, verbose=False)

    counts = {"OUTSIDE": 0, "GATE": 0, "INSIDE": 0}

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(heatmap, (cx, cy), 30, 5, -1)

        for zone_name, zone_coords in ZONES.items():
            if in_zone(cx, cy, zone_coords):
                counts[zone_name] += 1
                break

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Heatmap processing
    heatmap_blur = cv2.GaussianBlur(heatmap, (31, 31), 0)
    heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(np.uint8(heatmap_norm), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # Smoothed gate count
    gate_history.append(counts["GATE"])
    gate_count = int(sum(gate_history) / len(gate_history))
    density = gate_count / MAX_GATE_CAPACITY

    # Risk logic
    if density < 0.6:
        risk = "LOW"
        risk_color = "🟢"
        action = "Normal flow"
    elif density < 0.85:
        risk = "MEDIUM"
        risk_color = "🟡"
        action = "Monitor crowd. Prepare control."
    else:
        risk = "HIGH"
        risk_color = "🔴"
        action = "STOP entry. Allow exit only!"

    # Update dashboard
    frame_placeholder.image(overlay, channels="BGR")

    outside_metric.metric("Outside Count", counts["OUTSIDE"])
    gate_metric.metric("Gate Count", gate_count)
    inside_metric.metric("Inside Count", counts["INSIDE"])

    risk_box.subheader(f"Risk Level: {risk_color} {risk}")
    action_box.write(f"**Action:** {action}")
