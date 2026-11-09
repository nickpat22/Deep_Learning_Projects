# EmotionVision AI

Real-time facial-expression classification using a Convolutional Neural Network (CNN), TensorFlow/Keras, OpenCV, and Streamlit.

The project has two stages: model training in Google Colab and local webcam inference through a Streamlit interface. Training is performed separately; the Streamlit app loads the saved model and does not retrain it at startup.

> **Scope note:** This project classifies visible facial-expression patterns. It does not determine a person's actual feelings, intentions, personality, or mental state. Predictions are estimates and can be incorrect.

## Features

- Face detection using OpenCV's Haar Cascade classifier
- Seven-class facial-expression classification using a trained CNN
- Live webcam preview with predicted class and confidence score
- Start/stop camera controls
- Separate Google Colab notebook for model training
- Saved Keras model reused for inference

## Emotion classes

The model's class order is stored in `models/labels.json` and currently contains:

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
| Language | Python |
| Deep learning | TensorFlow / Keras |
| Model | Convolutional Neural Network (CNN) |
| Face detection | OpenCV Haar Cascade |
| User interface | Streamlit |
| Image processing | OpenCV, NumPy |
| Training environment | Google Colab |

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

1. **Training:** The Colab notebook loads grayscale images from the FER-2013-style directory structure, resizes them to 48 × 48 pixels, normalizes pixel values, and trains a CNN.
2. **Export:** The notebook saves the trained model as `emotion_model.keras` and the class order as `labels.json`.
3. **Face detection:** The Streamlit application reads frames from the local webcam and detects faces using Haar Cascade.
4. **Preprocessing:** Each detected face is converted to grayscale, resized to 48 × 48, and normalized to the range 0–1.
5. **Inference:** The CNN predicts one of the seven expression classes. The application overlays the predicted class and confidence on the webcam frame.

### Model input and output

| Property | Value |
|---|---|
| Input shape | `(48, 48, 1)` |
| Input format | Grayscale image |
| Output shape | `(7,)` |
| Output activation | Softmax |
| Classes | 7 facial-expression categories |

## Run locally

### Prerequisites

- Windows, macOS, or Linux
- Python 3.11 recommended
- A working webcam
- Git

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

### 4. Start the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal. Keep the webcam connected and close other applications that may be using it.

## Train the model

Open `EmotionVisionAI_Colab_Training.ipynb` in Google Colab and run its cells in order. The notebook expects a dataset ZIP containing:

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

The folder names are used as class labels. After training, export the model and label files and place them in the local `models/` directory. The dataset itself is not required to run the inference application.

## Important deployment limitation

The current webcam implementation uses `cv2.VideoCapture(0)`. This captures video from the machine running the Streamlit Python process. It is suitable for local execution, but it will **not** access a visitor's webcam when hosted on a cloud server. Cloud deployment requires a browser-side camera/WebRTC implementation and appropriate hosting/network support.

## Limitations

- FER-2013 is a challenging dataset; class imbalance and image quality can affect predictions.
- Lighting, camera angle, occlusion, and face size can reduce detection and classification quality.
- Confidence scores are model outputs, not calibrated probabilities of a person's true emotional state.
- Haar Cascade may miss faces or produce false detections.
- The current implementation is intended for local webcam inference, not production or high-stakes use.

## Repository

[Deep_Learning_Projects](https://github.com/nickpat22/Deep_Learning_Projects)

## Author

**Nikunj Patidar**  
GitHub: [@nickpat22](https://github.com/nickpat22)
