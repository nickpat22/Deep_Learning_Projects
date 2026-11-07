# Reusable AI/ML Project Template — Vercel

A reusable template for deploying Python machine-learning projects on Vercel. It uses FastAPI for the Python API, a minimal browser interface, and MongoDB for prediction history.

The current example is live concrete crack detection using an existing TensorFlow CNN model.

## Architecture

```text
Browser camera
    |
    | JPEG frame (HTTP POST)
    v
Vercel Python Function / FastAPI
    |
    +--> core/preprocessing.py
    |
    +--> core/model.py
    |
    +--> core/prediction.py
    |
    +--> MongoDB prediction history
```

The browser captures a frame and sends it to the Python API every two seconds. The API runs inference and returns JSON. The browser updates the result. One prediction is saved to MongoDB approximately every ten seconds.

This uses ordinary HTTP requests, not a persistent webcam connection or WebSocket server.

## Folder structure

```text
Reusable-AI-ML-Vercel-Template/
├── api/
│   └── index.py              # Common FastAPI endpoints
├── core/
│   ├── model.py              # PROJECT-SPECIFIC: load your model
│   ├── preprocessing.py      # PROJECT-SPECIFIC: prepare model input
│   └── prediction.py         # PROJECT-SPECIFIC: run model prediction
├── database/
│   └── mongodb.py            # Common MongoDB operations
├── models/
│   └── crack_detection_model.keras
├── public/
│   ├── index.html            # Common browser interface
│   ├── style.css             # Common styling
│   └── script.js             # Common browser camera and API calls
├── .env.example
├── .gitignore
├── .python-version
├── requirements.txt
├── vercel.json
└── README.md
```

## Reuse this template for future projects

Keep the folder structure and common files. Normally replace only these three files:

1. `core/model.py` — load the new trained model.
2. `core/preprocessing.py` — convert incoming input into the format expected by the new model.
3. `core/prediction.py` — implement the new prediction and return a JSON-compatible dictionary.

Keep the response contract stable so the common frontend can display it:

```python
{
    "prediction": "YOUR_LABEL",
    "confidence": 95.0,
    "raw_model_output": 0.95
}
```

If the new project requires a different input method (text, tabular data, audio, etc.), change the relevant controls in `public/index.html` and request logic in `public/script.js`. The common FastAPI and MongoDB pattern can remain.

Update `PROJECT_NAME` for each deployment. Use a separate MongoDB collection or database if projects need isolated history.

## Current crack detection model

Place the already-trained Keras model at:

`models/crack_detection_model.keras`

The `.keras` file is loaded directly with `tf.keras.models.load_model()`; no conversion to HDF5 is required. The current preprocessing preserves the supplied webcam code's 128 × 128 resize, RGB conversion, float32 conversion, normalization by 255, and batch dimension.

### Class mapping correction

The supplied training script uses `flow_from_directory()` with folders named `Crack` and `NoCrack`. Keras sorts class names alphabetically:

- `Crack` = class index 0
- `NoCrack` = class index 1

The final Dense layer uses sigmoid, so the output corresponds to class index 1 (`NoCrack`). Thus:

- output >= 0.5: `NO CRACK`
- output < 0.5: `CRACK`

This is why the prediction mapping in this template is the reverse of the original standalone webcam script's displayed labels. Confirm it against the exact model and training class mapping before relying on predictions.

## Local setup (Windows)

Use Python 3.12 for the Vercel runtime parity.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your MongoDB URI. Do not commit `.env`.

PowerShell example:

```powershell
$env:MONGODB_URI="mongodb://127.0.0.1:27017/"
$env:MONGODB_DATABASE="ai_ml_projects"
$env:MONGODB_COLLECTION="prediction_history"
$env:PROJECT_NAME="crack-detection"
uvicorn api.index:app --reload
```

Open `http://127.0.0.1:8000`.

The browser webcam requires localhost or HTTPS and explicit camera permission.

## MongoDB setup

For local development, use a running local MongoDB server and the local URI.

For Vercel deployment, use MongoDB Atlas or another publicly reachable MongoDB service. `localhost` on Vercel refers to the temporary function environment, not your laptop.

Set these environment variables in Vercel:

- `MONGODB_URI`
- `MONGODB_DATABASE` (optional; defaults to `ai_ml_projects`)
- `MONGODB_COLLECTION` (optional; defaults to `prediction_history`)
- `PROJECT_NAME` (optional; defaults to `crack-detection`)
- `MODEL_PATH` (optional; defaults to `models/crack_detection_model.keras`)

Never commit database credentials.

## Deploy to Vercel

1. Push the project to a GitHub repository.
2. Sign in to Vercel and select **Add New → Project**.
3. Import the GitHub repository.
4. Keep the project root as the repository root.
5. Configure the environment variables listed above in Project Settings → Environment Variables.
6. Deploy.
7. Open the deployed HTTPS URL and grant camera permission.
8. Test `/api/health`, live prediction, and MongoDB history.

Vercel's Python runtime supports FastAPI. The Python version is selected by `.python-version`.

## TensorFlow / model deployment warning

TensorFlow is a substantial dependency. Vercel documents a 500 MB uncompressed standard Python Function bundle limit and a 4.5 MB request/response body limit. Vercel also documents a Large Functions beta with a higher package ceiling for eligible projects. Check the current Vercel plan, Fluid Compute eligibility, and Large Functions availability before deploying this TensorFlow model.

Do not assume the model will deploy just because a small FastAPI example deploys. Check the Vercel build logs and actual function bundle size. Keep webcam JPEG frames below 2 MB; this template rejects larger frames before inference.

A model can also exceed available memory or have cold-start latency. If the TensorFlow dependency plus model does not fit the available Vercel function resources, host the model inference API on a Python ML host and keep the frontend on Vercel. The folder structure and API contract can remain the same.

## API endpoints

- `GET /api/health` — health check.
- `POST /api/predict` — multipart form fields: `image` and optional `save_history`.
- `GET /api/history` — latest ten saved predictions.

## Interview explanation

- **FastAPI:** receives image frames and returns prediction results.
- **Model loading:** loads the trained CNN once per warm function instance.
- **Preprocessing:** applies the exact image transformations required by the model.
- **Prediction:** converts sigmoid output into the class label and confidence.
- **MongoDB:** stores selected prediction records with UTC timestamps.
- **Vercel:** deploys the Python API and serves the browser interface.

## Limitations

- This is an inference application; it does not retrain the model.
- Serverless functions are stateless between invocations. MongoDB stores durable history.
- Browser camera access requires HTTPS or localhost and user permission.
- Live predictions are repeated HTTP requests at a modest frame rate, not full-FPS video inference.
- The system is a demonstration and is not a certified structural safety inspection tool.
