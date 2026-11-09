# EmotionVision AI — CNN + Streamlit

A simple facial-expression recognition project using a CNN trained on FER-2013.
Training is performed in Google Colab. The Streamlit app loads the saved model;
it does not train the model when the app starts.

> Important: The model classifies visible facial-expression patterns (for example,
> happy, sad, angry). It cannot reliably identify a person's actual internal
> sentiment or emotional state.

## 1. Project structure

```text
EmotionVisionAI_Streamlit/
│
├── app.py
├── requirements.txt
├── README.md
│
└── models/
    ├── emotion_model.keras   # Copy from Colab after training
    └── labels.json           # Copy from Colab after training
```

## 2. Train the model in Google Colab

1. Open the included `EmotionVisionAI_Colab_Training.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. When prompted, upload the dataset ZIP. The supplied archive is expected to
   contain `train/<class_name>/...` and `test/<class_name>/...`.
4. The notebook creates:
   - `models/emotion_model.keras`
   - `models/labels.json`
   - `models/accuracy.png`
   - `models/loss.png`
5. At the end, download `emotionvision_model_files.zip`.
6. Extract it and copy `emotion_model.keras` and `labels.json` into this
   project's `models/` folder.

The notebook uses the directory names as class labels, so the training and
prediction label order remains consistent.

## 3. Run Streamlit locally

Use Python 3.10 or 3.11 in a fresh virtual environment.

### Windows PowerShell

```powershell
cd EmotionVisionAI_Streamlit
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal. Allow camera access in the browser.
For webcam access, use localhost or deploy with HTTPS.

## 4. Deploy

Push the project to GitHub with the two trained model files included, then deploy
the repository on a Streamlit-compatible hosting service. Configure the app to
run:

```bash
streamlit run app.py
```

The hosting platform must support WebRTC camera connections. Some hosted
environments or networks block WebRTC; test camera access after deployment.

## 5. Common issues

- **Model file missing:** Check that both files are inside `models/`, not nested
  in another extracted folder.
- **Camera does not start:** Allow browser camera permission; test on localhost
  or HTTPS. WebRTC can be blocked by restrictive networks.
- **TensorFlow installation fails:** Use Python 3.10 or 3.11 and a clean virtual
  environment. Do not mix incompatible TensorFlow installations.
- **Poor predictions:** FER-2013 is a challenging dataset. Lighting, face angle,
  occlusion, class imbalance, and dataset limitations affect results.
