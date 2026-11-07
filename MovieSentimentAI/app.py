import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="MovieSentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

VOCAB_SIZE = 10_000
MAX_LENGTH = 200

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0b1020; color: #edf1ff; }
    [data-testid="stSidebar"] { background: #111a30; border-right: 1px solid #263451; }
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 20px;
        background: linear-gradient(120deg, #17264a 0%, #24205a 55%, #3a2060 100%);
        border: 1px solid #39466f; margin-bottom: 1.2rem;
    }
    .hero h1 { font-family: 'Manrope', sans-serif; color: #fff; font-size: 2.25rem; margin: 0; }
    .hero p { color: #c5cbea; margin: .45rem 0 0; font-size: 1rem; }
    .metric-card {
        background: #141e34; border: 1px solid #293957;
        border-radius: 16px; padding: 1rem 1.15rem; min-height: 105px;
    }
    .metric-label { color: #9eaccb; font-size: .82rem; }
    .metric-value { color: #fff; font-family: 'Manrope', sans-serif; font-size: 1.55rem; font-weight: 800; margin-top: .25rem; }
    .muted { color: #aab6d2; }
    .result-positive { background: #103c35; border: 1px solid #247e69; border-radius: 16px; padding: 1.1rem; }
    .result-negative { background: #421f31; border: 1px solid #a34b68; border-radius: 16px; padding: 1.1rem; }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #7658e8, #a05be8);
        border: 0; color: white; font-weight: 700; border-radius: 10px;
    }
    div[data-testid="stProgressBar"] > div > div { background: #9872f0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Model training (cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def train_lstm_model(training_samples: int, epochs: int):
    """Train the LSTM model once per selected configuration."""
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    x_train = x_train[:training_samples]
    y_train = y_train[:training_samples]

    x_train_pad = pad_sequences(x_train, maxlen=MAX_LENGTH)
    x_test_pad = pad_sequences(x_test, maxlen=MAX_LENGTH)

    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_LENGTH),
        LSTM(64, dropout=0.2, recurrent_dropout=0.0),
        Dense(32, activation="relu"),
        Dropout(0.25),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train_pad,
        np.asarray(y_train),
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
        verbose=0,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test_pad, np.asarray(y_test), verbose=0
    )
    word_index = imdb.get_word_index()
    return model, word_index, float(test_accuracy), history.history


def encode_review(review: str, word_index: dict) -> np.ndarray:
    """Convert ordinary text into the integer sequence used by the IMDB model."""
    tokens = text_to_word_sequence(review)
    sequence = [1]  # IMDB start token
    for token in tokens:
        index = word_index.get(token)
        sequence.append(index + 3 if index is not None and index < VOCAB_SIZE - 3 else 2)
    return pad_sequences([sequence], maxlen=MAX_LENGTH)


def predict_sentiment(review: str, model, word_index: dict):
    encoded = encode_review(review, word_index)
    probability = float(model.predict(encoded, verbose=0)[0][0])
    label = "Positive" if probability >= 0.5 else "Negative"
    confidence = probability if label == "Positive" else 1 - probability
    return label, probability, confidence


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## 🎬 MovieSentiment AI")
    st.caption("LSTM-powered review classifier")
    st.divider()
    st.markdown("### Model settings")
    training_samples = st.select_slider(
        "Training reviews",
        options=[5000, 10000, 15000, 25000],
        value=15000,
        format_func=lambda x: f"{x:,}",
    )
    epochs = st.select_slider("Training epochs", options=[1, 2, 3], value=3)
    st.caption("More training data may improve results but takes longer.")
    st.divider()
    st.markdown("**Architecture**")
    st.markdown("`Embedding → LSTM → Dense → Sigmoid`")
    st.markdown("**Dataset**")
    st.markdown("IMDB movie reviews")
    st.markdown("**Vocabulary**")
    st.markdown("10,000 most frequent words")

# ----------------------------
# Main interface
# ----------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Understand the emotion behind every review.</h1>
      <p>Analyze movie reviews with a recurrent neural network trained on the IMDB dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = None

model_col, data_col, arch_col = st.columns(3)
with model_col:
    status = "Trained" if st.session_state.model_bundle else "Not trained"
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">MODEL STATUS</div>'
        f'<div class="metric-value">{status}</div></div>',
        unsafe_allow_html=True,
    )
with data_col:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">MAX VOCABULARY</div>'
        '<div class="metric-value">10,000 words</div></div>',
        unsafe_allow_html=True,
    )
with arch_col:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">SEQUENCE LENGTH</div>'
        '<div class="metric-value">200 tokens</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Review analyzer")
    st.write("Enter a movie review or load an example to analyze its sentiment.")
    examples = {
        "Choose an example…": "",
        "Positive example": "A wonderful film with excellent acting, a moving story, and beautiful cinematography. I enjoyed every minute.",
        "Negative example": "The plot was painfully boring, the acting felt weak, and the ending was disappointing. I would not recommend it.",
        "Mixed example": "The performances were great, but the story was predictable and the pacing was slow.",
    }
    selected_example = st.selectbox("Try a sample review", list(examples.keys()))
    review_text = st.text_area(
        "Your movie review",
        value=examples[selected_example],
        height=170,
        placeholder="Type your review here…",
        help="The model uses the 10,000-word IMDB vocabulary. Unrecognized words are treated as unknown tokens.",
    )

    train_clicked = st.button(
        "⚡ Train LSTM model",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.model_bundle is not None,
    )

    if train_clicked:
        with st.spinner("Training the LSTM model. The first run downloads the IMDB dataset and may take a few minutes…"):
            try:
                bundle = train_lstm_model(training_samples, epochs)
                st.session_state.model_bundle = bundle
                st.success("Training completed. The model is ready for predictions.")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.info("Check the deployment logs and confirm that the app has enough memory.")

    if st.session_state.model_bundle is not None:
        if st.button("Reset model / change training settings", use_container_width=True):
            st.session_state.model_bundle = None
            st.rerun()

    predict_clicked = st.button(
        "Analyze sentiment",
        use_container_width=True,
        disabled=st.session_state.model_bundle is None,
    )

    if predict_clicked:
        if not review_text.strip():
            st.warning("Enter a review before running the prediction.")
        else:
            model, word_index, test_accuracy, history = st.session_state.model_bundle
            label, probability, confidence = predict_sentiment(review_text, model, word_index)
            st.session_state.last_prediction = {
                "label": label,
                "probability": probability,
                "confidence": confidence,
            }

with right:
    st.subheader("Prediction")
    result = st.session_state.get("last_prediction")
    if result:
        is_positive = result["label"] == "Positive"
        result_class = "result-positive" if is_positive else "result-negative"
        symbol = "😊" if is_positive else "😞"
        st.markdown(
            f'<div class="{result_class}"><div style="font-size:2rem">{symbol}</div>'
            f'<div style="font-size:.85rem;opacity:.8">PREDICTED SENTIMENT</div>'
            f'<div style="font-size:1.8rem;font-weight:800">{result["label"]}</div>'
            f'<div style="margin-top:.4rem">Confidence: {result["confidence"]:.1%}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("Positive probability")
        st.progress(result["probability"])
        st.caption(f"{result['probability']:.1%} probability of a positive review")
        st.caption("Confidence is the probability assigned to the predicted class.")
    elif st.session_state.model_bundle is None:
        st.info("Train the model first. Prediction controls will become available after training.")
    else:
        st.info("Your prediction will appear here.")

st.divider()
st.subheader("Model performance")

if st.session_state.model_bundle is None:
    st.caption("Train the model to view its test accuracy and training history.")
else:
    model, word_index, test_accuracy, history = st.session_state.model_bundle
    m1, m2, m3 = st.columns(3)
    m1.metric("Test accuracy", f"{test_accuracy:.1%}")
    m2.metric("Training epochs", len(history.get("accuracy", [])))
    m3.metric("Training reviews", f"{training_samples:,}")

    chart_data = pd.DataFrame({
        "Training accuracy": history.get("accuracy", []),
        "Validation accuracy": history.get("val_accuracy", []),
    })
    st.line_chart(chart_data, y=["Training accuracy", "Validation accuracy"])
    st.caption("Accuracy is measured on the held-out IMDB test split. Actual results may vary with training settings.")

with st.expander("How does the model work?"):
    st.markdown(
        """
        1. **Text preprocessing:** Reviews are converted into integer tokens using the IMDB word index.
        2. **Padding:** Each review is adjusted to 200 tokens.
        3. **Embedding:** Each token is represented by a 32-dimensional vector.
        4. **LSTM:** The recurrent layer learns sequential patterns in the review.
        5. **Sigmoid output:** The final layer produces a positive-class probability. Values at or above 0.5 are classified as Positive.
        """
    )

st.caption("Educational demo • TensorFlow / Keras • Streamlit • No database")
