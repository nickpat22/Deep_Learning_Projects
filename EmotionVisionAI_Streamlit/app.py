
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "emotion_model.keras"
LABELS_PATH = BASE_DIR / "models" / "labels.json"


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EmotionVision AI",
    page_icon="😊",
    layout="centered"
)

st.title("😊 EmotionVision AI")
st.write("Real-Time Facial Emotion Detection")


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(LABELS_PATH, "r") as file:
        labels = json.load(file)

    return model, labels


try:
    model, emotion_labels = load_model()

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# --------------------------------------------------
# 4. FACE DETECTOR
# --------------------------------------------------

cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    st.error("Unable to load face detector.")
    st.stop()


# --------------------------------------------------
# 5. SESSION STATE
# --------------------------------------------------

if "camera" not in st.session_state:
    st.session_state.camera = None

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

if "last_predictions" not in st.session_state:
    st.session_state.last_predictions = {}


# --------------------------------------------------
# 6. CAMERA UI AND PROCESSING
# --------------------------------------------------

@st.fragment(run_every=0.15)
def camera_display():

    # Camera controls
    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "▶ Start Camera",
            disabled=st.session_state.camera_running,
            use_container_width=True
        ):

            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if camera.isOpened():

                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                st.session_state.camera = camera
                st.session_state.camera_running = True
                st.session_state.frame_count = 0
                st.session_state.last_predictions = {}

            else:
                camera.release()
                st.error("Unable to access webcam.")

    with col2:

        if st.button(
            "⏹ Stop Camera",
            disabled=not st.session_state.camera_running,
            use_container_width=True
        ):

            if st.session_state.camera is not None:
                st.session_state.camera.release()

            st.session_state.camera = None
            st.session_state.camera_running = False
            st.session_state.last_predictions = {}

    # Show instructions when camera is stopped
    if not st.session_state.camera_running:

        st.info("Click Start Camera to begin detection.")
        return

    camera = st.session_state.camera

    if camera is None or not camera.isOpened():

        st.error("Camera connection lost.")
        return

    # Read webcam frame
    success, frame = camera.read()

    if not success:
        st.warning("Unable to read webcam frame.")
        return

    # Resize frame for smaller display and faster processing
    frame = cv2.resize(frame, (560, 420))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40)
    )

    st.session_state.frame_count += 1

    # Run CNN inference every second frame
    for index, (x, y, w, h) in enumerate(faces):

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (48, 48))

        face = face.astype("float32") / 255.0

        face = np.reshape(face, (1, 48, 48, 1))

        if st.session_state.frame_count % 2 == 0:

            prediction = model.predict(face, verbose=0)[0]

            emotion_index = int(np.argmax(prediction))

            emotion = emotion_labels[emotion_index]

            confidence = float(prediction[emotion_index]) * 100

            st.session_state.last_predictions[index] = (
                emotion,
                confidence
            )

        result = st.session_state.last_predictions.get(index)

        if result is not None:

            emotion, confidence = result

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # Draw emotion label
            label = f"{emotion.upper()} {confidence:.1f}%"

            cv2.putText(
                frame,
                label,
                (x, max(y-10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

    # Center the smaller webcam preview
    left, center, right = st.columns([1, 4, 1])

    with center:

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            frame_rgb,
            channels="RGB",
            width=560
        )

        st.caption(f"Faces detected: {len(faces)}")


# --------------------------------------------------
# 7. DISPLAY CAMERA
# --------------------------------------------------

camera_display()


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "EmotionVision AI | CNN-based Facial Emotion Recognition"
)
