from core.preprocessing import preprocess_image


def predict_image(model, image_bytes):
    """
    Project-specific prediction logic.

    Training used flow_from_directory() with folders named Crack and NoCrack.
    Keras sorts class names alphabetically:
        Crack   -> 0
        NoCrack -> 1

    The sigmoid output therefore represents NoCrack probability.
    """
    image_array = preprocess_image(image_bytes)
    raw_prediction = float(model.predict(image_array, verbose=0)[0][0])

    if raw_prediction >= 0.5:
        label = "NO CRACK"
        confidence = raw_prediction * 100
    else:
        label = "CRACK"
        confidence = (1.0 - raw_prediction) * 100

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "raw_model_output": round(raw_prediction, 6),
    }
