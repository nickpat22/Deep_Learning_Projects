from functools import lru_cache
from pathlib import Path
import os

import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "crack_detection_model.keras"


@lru_cache(maxsize=1)
def get_model():
    """
    Load the trained model once per warm serverless instance.

    For a new project, this is one of the project-specific files to replace.
    Set MODEL_PATH in the deployment environment to use a different location.
    """
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Place your trained model in the models folder or set MODEL_PATH."
        )

    return tf.keras.models.load_model(model_path)
