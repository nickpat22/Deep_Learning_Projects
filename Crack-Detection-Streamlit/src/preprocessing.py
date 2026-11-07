import cv2
import numpy as np


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preserve the preprocessing used by the original application:
    decode BGR -> RGB, resize to 128x128, float32, divide by 255,
    then add the batch dimension.
    """
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if bgr_image is None:
        raise ValueError("Could not decode the supplied image. Use a valid JPG or PNG.")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, (128, 128))
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)
