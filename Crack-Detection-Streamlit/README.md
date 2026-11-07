# Concrete Crack Detection — Streamlit

A pure-Python Streamlit application for concrete surface crack classification using an existing TensorFlow CNN model and optional MongoDB prediction history.

## Features

- Upload JPG/JPEG/PNG images.
- Capture an image with Streamlit camera input.
- Run inference using the existing trained Keras model.
- Display CRACK / NO CRACK, confidence, and raw sigmoid output.
- Optionally store prediction history in MongoDB.
- View and export recent prediction history as CSV.

## Project structure

```text
Crack-Detection-Streamlit/
├── app.py
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── prediction.py
│   └── database.py
├── models/
│   └── crack_detection_model.keras
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Model and preprocessing

The original preprocessing is retained:
1. Decode image bytes using OpenCV.
2. Convert BGR to RGB.
3. Resize to 128 × 128.
4. Convert to `float32`.
5. Divide pixel values by 255.
6. Add a batch dimension.

The original training folders were `Crack` and `NoCrack`. Keras alphabetical class ordering is `Crack=0`, `NoCrack=1`. The sigmoid output therefore represents the probability of `NoCrack`:
- Output >= 0.5: `NO CRACK`
- Output < 0.5: `CRACK`

Verify this mapping against the exact training run and model before relying on predictions.

## Local setup (Windows PowerShell)

Use Python 3.12, matching the original project runtime target.

```powershell
cd path\to\Crack-Detection-Streamlit
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the application:

```powershell
streamlit run app.py
```

Streamlit will print the local URL, usually `http://localhost:8501`.

## MongoDB configuration

MongoDB is optional for inference. Without a URI, image predictions still work but are not persisted.

For local development, create `.streamlit/secrets.toml` using `.streamlit/secrets.toml.example` as a template:

```toml
MONGODB_URI = "mongodb+srv://<username>:<password>@<cluster-url>/"
MONGODB_DATABASE = "ai_ml_projects"
MONGODB_COLLECTION = "prediction_history"
PROJECT_NAME = "crack-detection"
```

Never commit the real `secrets.toml` file. It is ignored by `.gitignore`.

Alternatively, set `MONGODB_URI`, `MONGODB_DATABASE`, `MONGODB_COLLECTION`, and `PROJECT_NAME` as environment variables.

## Deploy on Streamlit Community Cloud

1. Push this project folder to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**.
4. Choose the repository and branch.
5. Set the main file path to `Crack-Detection-Streamlit/app.py` if this folder is nested in a larger repository. If the app folder is the repository root, use `app.py`.
6. Under **Advanced settings → Secrets**, add the TOML values shown above.
7. Deploy and review the application logs.

TensorFlow is a large dependency. A successful local install does not guarantee the cloud environment has sufficient build time, memory, or disk capacity. If Community Cloud cannot build or run TensorFlow 2.19 with this model, use a host with suitable Python memory and dependency resources. Do not remove TensorFlow or change the model preprocessing as a blind workaround.

## Limitations

- This is image-by-image inference, not continuous video streaming.
- This application does not train or fine-tune the CNN.
- Prediction history requires a reachable MongoDB instance.
- This is a demonstration classifier, not a certified structural inspection system.
