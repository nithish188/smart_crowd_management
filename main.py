import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque

# ==============================
# LOAD MODEL
# ==============================
model = YOLO("yolov8n.pt")

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
# ZONES (x1, y1, x2, y2)
# ==============================
ZONE_OUTSIDE = (0, 0, 300, FRAME_HEIGHT)
ZONE_GATE    = (300, 0, 450, FRAME_HEIGHT)
ZONE_INSIDE  = (450, 0, FRAME_WIDTH, FRAME_HEIGHT)

ZONES = {
    "OUTSIDE": ZONE_OUTSIDE,
    "GATE": ZONE_GATE,
    "INSIDE": ZONE_INSIDE
}

# ==============================
# HELPER
# ==============================
def in_zone(cx, cy, zone):
    x1, y1, x2, y2 = zone
    return x1 < cx < x2 and y1 < cy < y2

# ==============================
# CROWD SETTINGS
# ==============================
MAX_GATE_CAPACITY = 3
gate_history = deque(maxlen=10)

last_risk = "LOW"
stable_counter = 0

# ==============================
# HEATMAP INIT
# ==============================
heatmap = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # ---- HEATMAP DECAY (IMPORTANT) ----
    heatmap *= 0.95

    # ---- YOLO DETECTION ----
    results = model(frame, classes=[0], conf=0.25, verbose=False)

    counts = {"OUTSIDE": 0, "GATE": 0, "INSIDE": 0}

    # ---- PROCESS DETECTIONS ----
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Strong heat contribution
        cv2.circle(heatmap, (cx, cy), 30, 5, -1)

        for zone_name, zone_coords in ZONES.items():
            if in_zone(cx, cy, zone_coords):
                counts[zone_name] += 1
                break

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # ==============================
    # HEATMAP PROCESSING
    # ==============================
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = np.uint8(heatmap_norm)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # ==============================
    # SMOOTHED COUNT
    # ==============================
    gate_history.append(counts["GATE"])
    gate_count = int(sum(gate_history) / len(gate_history))
    density = gate_count / MAX_GATE_CAPACITY

    # ==============================
    # RISK LOGIC
    # ==============================
    if density < 0.6:
        risk = "LOW"
        color = (0, 255, 0)
        action = "Normal flow"
    elif density < 0.85:
        risk = "MEDIUM"
        color = (0, 255, 255)
        action = "Monitor crowd. Prepare control."
    else:
        risk = "HIGH"
        color = (0, 0, 255)
        action = "STOP entry. Allow exit only!"

    # ---- STABILIZE RISK ----
    if risk != last_risk:
        stable_counter += 1
        if stable_counter < 5:
            risk = last_risk
        else:
            last_risk = risk
            stable_counter = 0
    else:
        stable_counter = 0

    # ==============================
    # DRAW ZONES
    # ==============================
    for zone_name, (x1, y1, x2, y2) in ZONES.items():
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(overlay, zone_name, (x1 + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ==============================
    # DISPLAY INFO
    # ==============================
    y = 30
    for zone, count in counts.items():
        cv2.putText(overlay, f"{zone}: {count}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30

    cv2.putText(overlay, f"Smoothed Gate Count: {gate_count}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.putText(overlay, f"RISK LEVEL: {risk}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.putText(overlay, f"ACTION: {action}",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ==============================
    # SHOW
    # ==============================
    cv2.imshow("Smart Crowd Monitoring – College Gate", overlay)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
