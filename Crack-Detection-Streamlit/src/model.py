from functools import lru_cache
import os
from pathlib import Path

import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "crack_detection_model.keras"


@lru_cache(maxsize=1)
def load_model():
    """Load the trained Keras model once per Streamlit server process."""
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Place crack_detection_model.keras in the models directory."
        )

    return tf.keras.models.load_model(model_path)
