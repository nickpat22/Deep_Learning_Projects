import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf


# --------------------------------------------------
# 1. PAGE CONFIGURATION AND PROJECT PATHS
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
st.write("Facial-expression classification using a trained CNN.")
st.caption(
    "Take a photo with your device camera or upload an existing image. "
    "The image is processed by the CNN on the app server."
)


# --------------------------------------------------
# 2. LOAD MODEL AND LABELS
# --------------------------------------------------

@st.cache_resource
def load_model_and_labels():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    with LABELS_PATH.open("r", encoding="utf-8") as file:
        loaded_labels = json.load(file)

    if loaded_model.input_shape[1:] != (48, 48, 1):
        raise ValueError(
            f"Expected model input shape (48, 48, 1), "
            f"received {loaded_model.input_shape}."
        )

    if loaded_model.output_shape[-1] != len(loaded_labels):
        raise ValueError(
            "The model output count does not match labels.json."
        )

    return loaded_model, loaded_labels


try:
    model, emotion_labels = load_model_and_labels()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# --------------------------------------------------
# 3. LOAD FACE DETECTOR
# --------------------------------------------------

cascade_file = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_file)

if face_detector.empty():
    st.error("OpenCV face detector could not be loaded.")
    st.stop()


# --------------------------------------------------
# 4. IMAGE PROCESSING AND EMOTION PREDICTION
# --------------------------------------------------

def analyze_image(image_bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("The uploaded image could not be read.")

    # Resize very large photos to keep processing memory reasonable.
    height, width = image.shape[:2]
    max_dimension = 1600

    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40),
    )

    results = []

    for face_number, (x, y, w, h) in enumerate(faces, start=1):
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = face.reshape(1, 48, 48, 1)

        predictions = model(face, training=False).numpy()[0]

        emotion_index = int(np.argmax(predictions))
        emotion = emotion_labels[emotion_index]
        confidence = float(predictions[emotion_index]) * 100

        results.append({
            "face": face_number,
            "emotion": emotion,
            "confidence": confidence,
        })

        # Draw face rectangle and predicted emotion.
        cv2.rectangle(
            output_image,
            (x, y),
            (x + w, y + h),
            (0, 220, 0),
            3,
        )

        label = f"{emotion.upper()} {confidence:.1f}%"
        text_y = max(y - 12, 25)

        cv2.putText(
            output_image,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    return image, output_image, results


# --------------------------------------------------
# 5. CAPTURE OR UPLOAD IMAGE
# --------------------------------------------------

st.subheader("1. Capture or upload a photo")

capture_tab, upload_tab = st.tabs(
    ["📸 Take a photo", "📁 Upload an image"]
)

with capture_tab:
    camera_photo = st.camera_input(
        "Take a photo using your device camera",
        key="emotionvision_camera_photo",
    )

with upload_tab:
    uploaded_photo = st.file_uploader(
        "Choose a photo",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG or PNG image containing one or more faces.",
        key="emotionvision_uploaded_photo",
    )


# --------------------------------------------------
# 6. ANALYZE SELECTED PHOTO
# --------------------------------------------------

selected_photo = camera_photo if camera_photo is not None else uploaded_photo

if selected_photo is not None:
    st.subheader("2. Emotion detection")

    try:
        with st.spinner("Detecting faces and analyzing expressions..."):
            original, processed, predictions = analyze_image(
                selected_photo.getvalue()
            )

        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original_rgb,
                caption="Original photo",
                use_container_width=True,
            )

        with col2:
            st.image(
                processed_rgb,
                caption="Face detection and predictions",
                use_container_width=True,
            )

        if predictions:
            st.success(f"Detected {len(predictions)} face(s).")
            st.subheader("Prediction results")

            for result in predictions:
                st.metric(
                    label=f"Face {result['face']} — {result['emotion'].title()}",
                    value=f"{result['confidence']:.2f}%",
                    help="CNN softmax score; not a calibrated measure of certainty.",
                )
        else:
            st.warning(
                "No face was detected. Try a clearer, front-facing photo "
                "with good lighting."
            )

    except Exception as error:
        st.error(f"Image processing failed: {error}")


# --------------------------------------------------
# 7. MODEL INFORMATION
# --------------------------------------------------

with st.expander("About this model"):
    st.write("**Model:** Convolutional Neural Network (CNN)")
    st.write("**Input:** 48 × 48 grayscale face image")
    st.write("**Output:** Seven facial-expression classes")
    st.write("**Classes:** " + ", ".join(emotion_labels))
    st.caption(
        "The model estimates visible facial-expression categories. "
        "It does not determine a person's actual internal emotional state."
    )

st.divider()
st.caption(
    "EmotionVision AI | TensorFlow · OpenCV · Streamlit"
)
