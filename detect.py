import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from collections import deque

# ==========================================
# Load Trained Model
# ==========================================

MODEL_PATH = "best_model.keras"

print("Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ==========================================
# Class Names
# ==========================================

CLASSES = [
    "With Mask",
    "Without Mask"
]

# ==========================================
# Colors
# ==========================================

GREEN = (0,255,0)
RED = (0,0,255)
YELLOW = (0,255,255)
CYAN = (255,255,0)
WHITE = (255,255,255)
BLUE = (255,120,0)
BLACK = (0,0,0)

# ==========================================
# MediaPipe Face Detector
# ==========================================

mp_face = mp.solutions.face_detection

face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.65
)

# ==========================================
# Webcam
# ==========================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

if not cap.isOpened():
    print("Cannot Open Webcam")
    exit()

# ==========================================
# FPS Variables
# ==========================================

prev_time = time.time()

# ==========================================
# Confidence History
# ==========================================

confidence_history = deque(maxlen=5)

# ==========================================
# Helper Function
# ==========================================

def draw_label(img,text,x,y,color):

    (w,h),baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        2
    )

    cv2.rectangle(
        img,
        (x,y-h-12),
        (x+w+10,y),
        color,
        -1
    )

    cv2.putText(
        img,
        text,
        (x+5,y-6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        WHITE,
        2
    )

# ==========================================
# Main Loop
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = face_detector.process(rgb)

    face_count = 0
        # ==========================================
    # Detect Faces
    # ==========================================

    if results.detections:

        face_count = len(results.detections)

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Keep coordinates inside frame
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            # Skip invalid faces
            if bw <= 0 or bh <= 0:
                continue

            face = frame[y:y+bh, x:x+bw]

            if face.size == 0:
                continue

            # ===============================
            # Prepare image for AI Model
            # ===============================

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face = cv2.resize(face, (224, 224))

            face = face.astype("float32") / 255.0

            face = np.expand_dims(face, axis=0)

            # ===============================
            # Prediction
            # ===============================

            prediction = model.predict(face, verbose=0)[0]

            class_id = np.argmax(prediction)

            confidence = float(prediction[class_id]) * 100

            # Smooth confidence
            confidence_history.append(confidence)

            confidence = sum(confidence_history) / len(confidence_history)

            label = CLASSES[class_id]

            color = GREEN if class_id == 0 else RED

            # Unknown if confidence is low
            if confidence < 60:
                label = "Unknown"
                color = YELLOW

            # ===============================
            # Draw Face Box
            # ===============================

            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                color,
                3
            )

            draw_label(
                frame,
                f"{label}  {confidence:.1f}%",
                x,
                y,
                color
            )

            # ===============================
            # Confidence Bar
            # ===============================

            bar_width = int((confidence / 100) * bw)

            cv2.rectangle(
                frame,
                (x, y + bh + 8),
                (x + bw, y + bh + 20),
                (60, 60, 60),
                -1
            )

            cv2.rectangle(
                frame,
                (x, y + bh + 8),
                (x + bar_width, y + bh + 20),
                color,
                -1
            )
                # ==========================================
    # FPS
    # ==========================================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # ==========================================
    # Dashboard
    # ==========================================

    cv2.rectangle(
        frame,
        (0, 0),
        (420, 120),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        frame,
        "AI Face Mask Detection",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        CYAN,
        2
    )

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        GREEN,
        2
    )

    cv2.putText(
        frame,
        f"Faces : {face_count}",
        (180, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        YELLOW,
        2
    )

    timestamp = time.strftime("%d-%m-%Y %H:%M:%S")

    cv2.putText(
        frame,
        timestamp,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WHITE,
        2
    )

    # ==========================================
    # Instructions
    # ==========================================

    cv2.putText(
        frame,
        "Press Q to Exit",
        (frame.shape[1]-200,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WHITE,
        2
    )

    # ==========================================
    # Display Window
    # ==========================================

    cv2.imshow(
        "AI Face Mask Detection",
        frame
    )

    # ==========================================
    # Exit
    # ==========================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ==========================================
# Cleanup
# ==========================================

cap.release()

cv2.destroyAllWindows()