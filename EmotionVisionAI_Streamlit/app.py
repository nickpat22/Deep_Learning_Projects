import json
import logging
from pathlib import Path

import av
import cv2
import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# --------------------------------------------------
# PAGE AND PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "emotion_model.keras"
LABELS_PATH = BASE_DIR / "models" / "labels.json"

st.set_page_config(page_title="EmotionVision AI", page_icon="😊", layout="centered")
st.title("😊 EmotionVision AI")
st.caption("Real-time facial-expression classification using a CNN.")

# --------------------------------------------------
# LOAD MODEL ONCE
# --------------------------------------------------
@st.cache_resource
def load_model_and_labels():
    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
        raise FileNotFoundError("Required files are missing from the models/ folder.")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    if model.input_shape[1:] != (48, 48, 1):
        raise ValueError(f"Expected model input (48, 48, 1), got {model.input_shape}.")
    if model.output_shape[-1] != len(labels):
        raise ValueError("Model output count does not match models/labels.json.")
    return model, labels

try:
    model, emotion_labels = load_model_and_labels()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

# --------------------------------------------------
# CLOUDFLARE TURN CREDENTIALS
# TURN is needed because STUN-only connectivity is not
# reliable on Streamlit Community Cloud.
# --------------------------------------------------
@st.cache_data(ttl=3000, show_spinner=False)
def get_ice_servers(key_id: str, api_token: str):
    url = (
        "https://rtc.live.cloudflare.com/v1/turn/keys/"
        f"{key_id}/credentials/generate-ice-servers"
    )
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_token}"},
        json={"ttl": 3600},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    servers = data.get("iceServers")
    if not servers:
        raise RuntimeError("Cloudflare returned no ICE servers.")
    return servers

try:
    turn_key_id = st.secrets["CLOUDFLARE_TURN_KEY_ID"]
    turn_api_token = st.secrets["CLOUDFLARE_TURN_API_TOKEN"]
    ice_servers = get_ice_servers(turn_key_id, turn_api_token)
    rtc_configuration = RTCConfiguration({"iceServers": ice_servers})
except KeyError:
    st.error(
        "Webcam relay is not configured. Add CLOUDFLARE_TURN_KEY_ID and "
        "CLOUDFLARE_TURN_API_TOKEN in Streamlit → App settings → Secrets. "
        "See the README deployment section."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not obtain TURN credentials from Cloudflare: {exc}")
    st.stop()

# --------------------------------------------------
# VIDEO PROCESSOR
# --------------------------------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.frame_count = 0
        self.last_prediction = None

    def recv(self, video_frame):
        frame = video_frame.to_ndarray(format="bgr24")
        frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.25, minNeighbors=5, minSize=(45, 45)
        )
        self.frame_count += 1

        if len(faces) == 0:
            self.last_prediction = None

        for (x, y, w, h) in faces:
            if self.frame_count % 3 == 1 or self.last_prediction is None:
                face = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                face = face.astype(np.float32) / 255.0
                face = face.reshape(1, 48, 48, 1)
                scores = self._model(face)
                idx = int(np.argmax(scores))
                self.last_prediction = (
                    emotion_labels[idx],
                    float(scores[idx]) * 100.0,
                )

            emotion, confidence = self.last_prediction
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{emotion.upper()} {confidence:.1f}%",
                (x, max(24, y-10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2,
            )

        return av.VideoFrame.from_ndarray(frame, format="bgr24")

    def _model(self, face):
        return model(face, training=False).numpy()[0]

# --------------------------------------------------
# BROWSER WEBCAM
# --------------------------------------------------
st.subheader("Live webcam")
st.info("Select START, allow camera access, and keep this page open while detecting.")

ctx = webrtc_streamer(
    key="emotionvision-cloud-webcam-v2",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480},
            "height": {"ideal": 360},
            "frameRate": {"ideal": 15, "max": 20},
            "facingMode": "user",
        },
        "audio": False,
    },
    async_processing=True,
)

if ctx.state.playing:
    st.success("Camera connected — emotion detection is running.")
elif ctx.state.signalling:
    st.info("Connecting webcam… If this takes more than 20 seconds, check camera permission and TURN configuration.")
else:
    st.caption("Camera is stopped. Press START to begin.")

with st.expander("About this model"):
    st.write("**Input:** 48 × 48 grayscale face image")
    st.write("**Output:** Seven facial-expression classes")
    st.write("**Classes:** " + ", ".join(emotion_labels))
    st.caption(
        "Predictions classify visible facial-expression patterns; they do not "
        "measure a person's actual internal emotional state."
    )

st.divider()
st.caption("EmotionVision AI | CNN · TensorFlow · OpenCV · WebRTC")
