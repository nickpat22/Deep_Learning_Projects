from src.preprocessing import preprocess_image


def predict_image(model, image_bytes: bytes) -> dict:
    """
    Training folders were Crack and NoCrack. Alphabetical Keras class order:
    Crack=0, NoCrack=1. With a sigmoid output, the scalar is NoCrack probability.
    """
    image_array = preprocess_image(image_bytes)
    raw_prediction = float(model.predict(image_array, verbose=0).reshape(-1)[0])

    if raw_prediction >= 0.5:
        label = "NO CRACK"
        confidence = raw_prediction * 100.0
    else:
        label = "CRACK"
        confidence = (1.0 - raw_prediction) * 100.0

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "raw_model_output": round(raw_prediction, 6),
    }
