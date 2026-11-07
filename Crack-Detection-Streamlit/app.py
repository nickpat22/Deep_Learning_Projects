from datetime import datetime
import pandas as pd
import streamlit as st

from src.model import load_model
from src.prediction import predict_image
from src.database import save_prediction, get_recent_predictions, database_is_configured

st.set_page_config(
    page_title="Concrete Crack Detection",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧱 Concrete Crack Detection")
st.caption("CNN-based image classification for concrete surface crack detection.")

with st.sidebar:
    st.header("Project")
    st.write("Upload an image or capture one with your camera.")
    st.divider()
    st.caption("Model input: 128 × 128 RGB image, normalized to [0, 1].")
    st.caption("Prediction classes: CRACK / NO CRACK.")

detect_tab, history_tab, about_tab = st.tabs(
    ["🔍 Detect", "🗂️ Prediction history", "ℹ️ About"]
)

with detect_tab:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Provide an image")
        input_method = st.radio(
            "Input method",
            ["Upload image", "Use camera"],
            horizontal=True,
        )
        uploaded = None
        if input_method == "Upload image":
            uploaded = st.file_uploader(
                "Choose a concrete surface image",
                type=["jpg", "jpeg", "png"],
                help="Use a clear image of a concrete surface.",
            )
        else:
            uploaded = st.camera_input("Capture a concrete surface")

        save_to_history = st.checkbox(
            "Save this prediction to MongoDB history",
            value=True,
        )

    with right:
        st.subheader("Prediction")
        if uploaded is None:
            st.info("Upload or capture an image to begin.")
        else:
            image_bytes = uploaded.getvalue()
            if len(image_bytes) > 2 * 1024 * 1024:
                st.error("Image exceeds the 2 MB application limit. Upload a smaller image.")
            elif not image_bytes:
                st.error("The supplied image is empty.")
            else:
                st.image(image_bytes, caption="Input image", use_container_width=True)
                if st.button("Run crack detection", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Loading model and analyzing image..."):
                            model = load_model()
                            result = predict_image(model, image_bytes)

                        st.session_state["last_prediction"] = result

                        if save_to_history:
                            try:
                                save_prediction(result)
                                st.session_state["history_saved"] = True
                            except Exception as db_error:
                                st.session_state["history_saved"] = False
                                st.session_state["history_error"] = str(db_error)
                        else:
                            st.session_state["history_saved"] = None

                    except Exception as error:
                        st.error(f"Prediction failed: {error}")

                result = st.session_state.get("last_prediction")
                if result:
                    st.divider()
                    if result["prediction"] == "CRACK":
                        st.error("⚠️ CRACK DETECTED")
                    else:
                        st.success("✅ NO CRACK DETECTED")

                    metric_col1, metric_col2 = st.columns(2)
                    metric_col1.metric("Prediction confidence", f'{result["confidence"]:.2f}%')
                    metric_col2.metric("Raw sigmoid output", f'{result["raw_model_output"]:.6f}')

                    st.progress(min(max(result["confidence"] / 100, 0.0), 1.0))

                    saved = st.session_state.get("history_saved")
                    if saved is True:
                        st.success("Prediction saved to MongoDB.")
                    elif saved is False:
                        st.warning(
                            "Prediction completed, but MongoDB history could not be saved. "
                            + st.session_state.get("history_error", "")
                        )
                    elif saved is None and save_to_history:
                        st.info("This prediction was not saved.")

with history_tab:
    st.subheader("Recent predictions")
    if not database_is_configured():
        st.info("MongoDB is not configured. Add MONGODB_URI to Streamlit secrets to enable history.")
    else:
        if st.button("Refresh history"):
            st.cache_data.clear()
        try:
            records = get_recent_predictions(limit=50)
            if records:
                frame = pd.DataFrame(records)
                if "created_at" in frame.columns:
                    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
                st.dataframe(frame, use_container_width=True, hide_index=True)
                csv_data = frame.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download history as CSV",
                    data=csv_data,
                    file_name="crack_detection_history.csv",
                    mime="text/csv",
                )
            else:
                st.info("No saved predictions yet.")
        except Exception as error:
            st.error(f"Could not load MongoDB history: {error}")

with about_tab:
    st.subheader("How this application works")
    st.markdown(
        """
        1. An image is uploaded or captured using the browser camera.
        2. The image is decoded, converted to RGB, resized to **128 × 128**,
           converted to `float32`, normalized by 255, and batched.
        3. The trained TensorFlow CNN returns a sigmoid value.
        4. The application maps the sigmoid output to **CRACK** or **NO CRACK**.
        5. If MongoDB is configured and saving is enabled, the result and UTC
           timestamp are stored in the prediction history collection.
        """
    )
    st.warning(
        "This is a demonstration classifier, not a certified structural inspection "
        "system. Do not use its output as the sole basis for safety decisions."
    )
