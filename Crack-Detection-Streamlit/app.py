
import hashlib

import streamlit as st

from src.model import load_model
from src.prediction import predict_image


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Concrete Crack Detection",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Application header
# --------------------------------------------------

st.title("🧱 Concrete Crack Detection")

st.caption(
    "CNN-based image classification for concrete surface crack detection."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("Project")

    st.write(
        "Upload an image or capture one with your camera."
    )

    st.divider()

    st.caption(
        "Model input: 128 × 128 RGB image, normalized to [0, 1]."
    )

    st.caption(
        "Prediction classes: CRACK / NO CRACK."
    )


# --------------------------------------------------
# Main interface
# --------------------------------------------------

detect_tab, about_tab = st.tabs(
    ["🔍 Detect", "ℹ️ About"]
)


# --------------------------------------------------
# Detection tab
# --------------------------------------------------

with detect_tab:

    left, right = st.columns(
        [1, 1],
        gap="large"
    )

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

            uploaded = st.camera_input(
                "Capture a concrete surface"
            )


    with right:

        st.subheader("Prediction")

        if uploaded is None:

            st.info(
                "Upload or capture an image to begin."
            )

        else:

            image_bytes = uploaded.getvalue()

            # Identify the current image so an old prediction
            # is not displayed against a newly selected image.
            image_hash = hashlib.sha256(
                image_bytes
            ).hexdigest()

            if len(image_bytes) > 2 * 1024 * 1024:

                st.error(
                    "Image exceeds the 2 MB application limit. "
                    "Upload a smaller image."
                )

            elif not image_bytes:

                st.error(
                    "The supplied image is empty."
                )

            else:

                st.image(
                    image_bytes,
                    caption="Input image",
                    use_container_width=True,
                )

                if st.button(
                    "Run crack detection",
                    type="primary",
                    use_container_width=True,
                ):

                    try:

                        with st.spinner(
                            "Loading model and analyzing image..."
                        ):

                            model = load_model()

                            result = predict_image(
                                model,
                                image_bytes
                            )

                        st.session_state["last_prediction"] = result

                        st.session_state["prediction_image_hash"] = (
                            image_hash
                        )

                    except Exception as error:

                        st.session_state.pop(
                            "last_prediction",
                            None
                        )

                        st.session_state.pop(
                            "prediction_image_hash",
                            None
                        )

                        st.error(
                            f"Prediction failed: {error}"
                        )


                # Display only the prediction belonging
                # to the currently selected image.

                result = st.session_state.get(
                    "last_prediction"
                )

                previous_hash = st.session_state.get(
                    "prediction_image_hash"
                )

                if result and previous_hash == image_hash:

                    st.divider()

                    if result["prediction"] == "CRACK":

                        st.error(
                            "⚠️ CRACK DETECTED"
                        )

                    else:

                        st.success(
                            "✅ NO CRACK DETECTED"
                        )


                    metric_col1, metric_col2 = st.columns(2)

                    metric_col1.metric(
                        "Prediction confidence",
                        f'{result["confidence"]:.2f}%'
                    )

                    metric_col2.metric(
                        "Raw sigmoid output",
                        f'{result["raw_model_output"]:.6f}'
                    )

                    st.progress(
                        min(
                            max(
                                result["confidence"] / 100,
                                0.0
                            ),
                            1.0
                        )
                    )


# --------------------------------------------------
# About tab
# --------------------------------------------------

with about_tab:

    st.subheader(
        "How this application works"
    )

    st.markdown(
        """
        1. An image is uploaded or captured using the browser camera.

        2. The image is decoded, converted to RGB, resized to
           **128 × 128**, converted to `float32`, normalized by 255,
           and batched.

        3. The trained TensorFlow CNN returns a sigmoid value.

        4. The application maps the sigmoid output to
           **CRACK** or **NO CRACK**.

        5. The prediction confidence is displayed to the user.
        """
    )

    st.warning(
        "This is a demonstration classifier, not a certified "
        "structural inspection system. Do not use its output as "
        "the sole basis for safety decisions."
    )
