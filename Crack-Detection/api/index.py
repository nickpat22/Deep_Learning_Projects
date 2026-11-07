import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.model import get_model
from core.prediction import predict_image
from database.mongodb import get_collection, save_prediction, get_recent_predictions

PUBLIC_DIR = ROOT_DIR / "public"
MAX_IMAGE_BYTES = 2 * 1024 * 1024

app = FastAPI(
    title="AI/ML Project API",
    description="Reusable FastAPI backend for Python machine-learning projects.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "AI/ML API is running"}


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    save_history: bool = Form(False),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a valid image.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Send an image smaller than 2 MB.")

    try:
        model = get_model()
        result = predict_image(model, image_bytes)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")

    if save_history:
        try:
            collection = get_collection()
            if collection is not None:
                save_prediction(collection, result)
                result["saved_to_database"] = True
            else:
                result["saved_to_database"] = False
        except Exception:
            # Prediction remains available if MongoDB is temporarily unavailable.
            result["saved_to_database"] = False
    else:
        result["saved_to_database"] = False

    return result


@app.get("/api/history")
def history():
    try:
        collection = get_collection()
        if collection is None:
            return {"enabled": False, "items": []}
        return {"enabled": True, "items": get_recent_predictions(collection, limit=10)}
    except Exception:
        return {"enabled": False, "items": []}
