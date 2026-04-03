import cv2
from ultralytics import YOLO
import time
import os

import numpy as np

# ---------- CONFIG ----------
MODEL_PATH = "d4.pt"
VIDEO_SOURCE = 0 # RTSP / video file / webcam
CAPTURE_COOLDOWN = 3  # seconds for auto-capture
SAVE_DIR = "mission2"

# ---------- CREATE SAVE DIR ----------
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- LOAD MODEL ----------
# Added safety check for CUDA availability
device = "cuda"
print(f" Loading model on {device}...")
model = YOLO(MODEL_PATH)
model.to(device)
CLASS_NAMES = model.names

# ---------- OPEN VIDEO ----------
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("⚠ Failed to open video source!")
    exit(1)

# ---------- FPS SETUP ----------
prev_time = 0
fps = 0

# ---------- CAPTURE TIMER ----------
last_capture_time = 0

# ---------- RECORDING SETUP (NEW) ----------
is_recording = False
video_writer = None

print("✅ System Ready.")
print("   - Press 's' to take a manual screenshot.")
print("   - Press 'r' to Toggle Video Recording.") # <--- Added instruction
print("   - Press 'q' or 'ESC' to quit.")

# ---------- MAIN LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # ---------- INFERENCE ----------
    results = model(frame, conf=0.4, verbose=False)

    # ---------- FPS ----------
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    object_detected = False

    # ---------- DRAW DETECTIONS ----------
    for r in results:
        for box in r.boxes:
            object_detected = True

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------- AUTO IMAGE CAPTURE LOGIC ----------
    if object_detected and (current_time - last_capture_time >= CAPTURE_COOLDOWN):
        filename = f"{SAVE_DIR}/auto_{int(current_time)}.jpg"
        cv2.imwrite(filename, frame)
        print(f"🤖 Auto-capture: {filename}")
        last_capture_time = current_time

    # ---------- VIDEO RECORDING LOGIC (NEW) ----------
    if is_recording and video_writer is not None:
        video_writer.write(frame)
        # Add a visual indicator (Red Circle) so you know it's recording
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) 

    # ---------- DISPLAY FPS ----------
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)

    # ---------- SHOW ----------
    resized_frame = cv2.resize(frame, (800, 600))
    cv2.imshow("Drone Tool Detection", resized_frame)

    # ---------- KEY CONTROLS ----------
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q') or key == 27:
        break
    
    # Manual Screenshot
    elif key == ord('s'):
        filename = f"{SAVE_DIR}/manual_{int(current_time)}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Manual Screenshot saved: {filename}")

    # Toggle Recording (NEW)
    elif key == ord('r'):
        if not is_recording:
            # Start Recording
            video_filename = f"{SAVE_DIR}/video_{int(current_time)}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Note: We use 640x480 because that's what you set cap to
            video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))
            is_recording = True
            print(f"🎥 Recording STARTED: {video_filename}")
        else:
            # Stop Recording
            is_recording = False
            if video_writer:
                video_writer.release()
                video_writer = None
            print("⏹️ Recording STOPPED.")

# Cleanup writer if still open
if video_writer:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()