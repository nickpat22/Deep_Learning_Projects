# EmotionVision AI

🌐 **Live Website:** https://emotionvisionai.streamlit.app/

A facial-expression classification web application built with Python, Streamlit, OpenCV, TensorFlow, and a trained Convolutional Neural Network (CNN). Users can take a photo with their device camera or upload an existing image. The application detects faces and predicts an expression class for each detected face.

> **Important:** This project estimates visible facial-expression categories. It does not determine a person's actual feelings, intentions, personality, or mental state. Predictions may be inaccurate and should not be used for high-stakes decisions.

## Features

- Capture a photo directly using the browser's camera through Streamlit's camera input
- Upload JPG, JPEG, and PNG images
- Detect multiple faces in a single image using OpenCV Haar Cascade
- Classify each detected face into one of seven expression categories using a trained CNN
- Display the original image and the processed image with face bounding boxes and predictions
- Show the model's confidence score for each detected face
- Reuse a saved Keras model; model training is separate from application inference

## Supported expression classes

The class order is stored in `models/labels.json`:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Technology stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning framework | TensorFlow / Keras |
| Model architecture | Convolutional Neural Network (CNN) |
| Face detection | OpenCV Haar Cascade |
| Image processing | OpenCV, NumPy |
| Web application | Streamlit |
| Model training environment | Google Colab |

## Project structure

```text
EmotionVisionAI_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── EmotionVisionAI_Colab_Training.ipynb
└── models/
    ├── emotion_model.keras
    └── labels.json
```

## How it works

1. **Capture or upload:** The user takes a photo using the browser camera or selects an existing JPG, JPEG, or PNG image.
2. **Image decoding:** OpenCV decodes the submitted image and resizes very large images to a maximum dimension of 1600 pixels.
3. **Face detection:** OpenCV's Haar Cascade detects face regions in the image.
4. **Preprocessing:** Each detected face is converted to grayscale, resized to 48 × 48 pixels, and normalized to the range 0–1.
5. **CNN inference:** The trained Keras model predicts one of the seven expression classes for each detected face.
6. **Result display:** The application shows the original photo, the annotated photo with bounding boxes and labels, and the prediction score for each face.

## Model details

| Property | Value |
|---|---|
| Saved model | `models/emotion_model.keras` |
| Input shape | `(48, 48, 1)` |
| Input format | Grayscale image |
| Output shape | `(7,)` |
| Output activation | Softmax |
| Number of classes | 7 |

The displayed confidence is the model's softmax score. It is not necessarily a calibrated probability or a measure of certainty about a person's actual emotional state.

## Run locally

### Prerequisites

- Python 3.11
- Git
- A webcam if you want to take a photo directly in the browser

### 1. Clone the repository

```bash
git clone https://github.com/nickpat22/Deep_Learning_Projects.git
cd Deep_Learning_Projects/EmotionVisionAI_Streamlit
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal.

## Deploy on Streamlit Community Cloud

The application is deployed at:

**https://emotionvisionai.streamlit.app/**

To deploy your own copy:

1. Fork or push this repository to your GitHub account.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app using your repository.
4. Set the branch to `main`.
5. Set the main file path to `EmotionVisionAI_Streamlit/app.py`.
6. Select **Python 3.11** in the deployment settings.
7. Deploy the app.

The application uses Streamlit's photo capture and file uploader components. It does not require WebRTC, a TURN server, Cloudflare credentials, or paid webcam-relay services.

## Train the model

The notebook `EmotionVisionAI_Colab_Training.ipynb` contains the model-training workflow. Open it in Google Colab and follow its cells. The training notebook expects a dataset arranged into class-specific training and testing folders, for example:

```text
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
└── test/
    ├── angry/
    ├── disgust/
    └── ...
```

The saved model and label file are required for inference. The training dataset is not needed to run the deployed application.

## Limitations

- Face detection may fail with poor lighting, unusual camera angles, occlusion, or small faces.
- Haar Cascade can produce false detections or miss faces.
- Model predictions depend on the training data and may not generalize to all people or environments.
- The confidence score is a model output, not a verified measure of a person's internal emotional state.
- The app analyzes submitted photos; it does not perform continuous real-time video emotion tracking.

## Repository

[Deep_Learning_Projects](https://github.com/nickpat22/Deep_Learning_Projects)

## Author

**Nikunj Patidar**  
GitHub: [@nickpat22](https://github.com/nickpat22)
