import json
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)


# --------------------------------------------------
# 1. PROJECT PATHS AND PAGE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "emotion_model.keras"
LABELS_PATH = BASE_DIR / "models" / "labels.json"

st.set_page_config(
    page_title="EmotionVision AI",
    page_icon="😊",
    layout="centered",
)

st.title("😊 EmotionVision AI")
st.write("Real-time facial-expression classification using a CNN.")


# --------------------------------------------------
# 2. LOAD MODEL ONCE
# --------------------------------------------------

@st.cache_resource
def load_emotion_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        loaded_labels = json.load(file)

    if loaded_model.input_shape[1:] != (48, 48, 1):
        raise ValueError(
            f"Unexpected model input shape: {loaded_model.input_shape}"
        )

    if loaded_model.output_shape[-1] != len(loaded_labels):
        raise ValueError(
            "The number of model outputs does not match labels.json."
        )

    return loaded_model, loaded_labels


try:
    model, emotion_labels = load_emotion_model()
except Exception as error:
    st.error(f"Could not load the emotion model: {error}")
    st.stop()


# --------------------------------------------------
# 3. WEBRTC VIDEO PROCESSOR
# --------------------------------------------------

class EmotionProcessor(VideoProcessorBase):
    def __init__(self, trained_model, labels):
        self.model = trained_model
        self.labels = labels
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )
        self.frame_count = 0
        self.last_predictions = {}

    def recv(self, video_frame):
        # Convert the incoming browser webcam frame to OpenCV format.
        frame = video_frame.to_ndarray(format="bgr24")

        # Keep processing resolution modest for faster inference.
        frame = cv2.resize(frame, (560, 420))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(40, 40),
        )

        self.frame_count += 1

        for index, (x, y, w, h) in enumerate(faces):
            # Run CNN inference every third frame to reduce CPU load.
            if self.frame_count % 3 == 0 or index not in self.last_predictions:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = face.astype("float32") / 255.0
                face = np.expand_dims(face, axis=(0, -1))

                probabilities = self.model(face, training=False).numpy()[0]
                emotion_index = int(np.argmax(probabilities))

                self.last_predictions[index] = (
                    self.labels[emotion_index],
                    float(probabilities[emotion_index]) * 100,
                )

            emotion, confidence = self.last_predictions[index]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            label = f"{emotion.upper()} {confidence:.1f}%"
            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

        # Remove cached predictions for faces that are no longer detected.
        self.last_predictions = {
            index: prediction
            for index, prediction in self.last_predictions.items()
            if index < len(faces)
        }

        return av.VideoFrame.from_ndarray(frame, format="bgr24")


# --------------------------------------------------
# 4. BROWSER WEBCAM STREAM
# --------------------------------------------------

st.subheader("Live webcam")

st.info(
    "Allow camera permission in your browser, then press START. "
    "Press STOP to end the webcam stream."
)

# STUN helps the browser and hosted server discover a network path.
# Some restrictive networks may additionally require a TURN server.
rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

left, center, right = st.columns([1, 3, 1])

with center:
    webrtc_ctx = webrtc_streamer(
        key="emotionvision-browser-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=lambda: EmotionProcessor(
            model,
            emotion_labels,
        ),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "facingMode": "user",
            },
            "audio": False,
        },
        async_processing=True,
    )

if webrtc_ctx.state.playing:
    st.success("Webcam is connected. Emotion detection is running.")
else:
    st.caption("Camera is currently stopped.")


# --------------------------------------------------
# 5. MODEL INFORMATION
# --------------------------------------------------

with st.expander("About this model"):
    st.write("**Input:** 48 × 48 grayscale face image")
    st.write("**Output:** Seven facial-expression classes")
    st.write("**Classes:** " + ", ".join(emotion_labels))
    st.write(
        "The model estimates visible facial-expression categories. "
        "It does not determine a person's actual internal emotional state."
    )

st.divider()
st.caption("EmotionVision AI | CNN · TensorFlow · OpenCV · Streamlit")
