import cv2
import numpy as np


def preprocess_image(image_bytes):
    """
    Decode uploaded webcam frame and apply the original CNN preprocessing:
    BGR -> RGB, resize to 128x128, float32, divide by 255, add batch dimension.

    Replace this function when a future project's model expects different input.
    """
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if bgr_image is None:
        raise ValueError("Could not decode the supplied image.")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, (128, 128))
    image_array = np.array(resized_image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array
