# MovieSentiment AI

A simple Streamlit web application that classifies movie reviews as **Positive** or **Negative** using an LSTM neural network trained on the Keras IMDB dataset.

## Features

- Review input with sample reviews
- LSTM-based binary sentiment classification
- Positive-class probability and prediction confidence
- Test accuracy and training/validation accuracy chart
- Train only when requested
- No database or external API required
- Streamlit caching avoids repeating training for the same settings during the app's runtime

## Tech stack

- Python 3.11
- Streamlit
- TensorFlow / Keras
- IMDB movie review dataset
- NumPy and Pandas

## Project structure

```text
MovieSentimentAI/
├── app.py
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

## Run locally

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start Streamlit

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this project.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub and select **Create app**.
4. Select your repository, branch, and `app.py` as the main file.
5. Deploy.

The first training run downloads the IMDB dataset. It can take several minutes and consumes memory/CPU. Start with 5,000 or 10,000 training reviews if the free deployment environment struggles.

## Model pipeline

```text
Movie review text
       |
       v
IMDB word-index tokenization
       |
       v
Sequence padding (200 tokens)
       |
       v
Embedding layer (32 dimensions)
       |
       v
LSTM layer (64 units)
       |
       v
Dense + Sigmoid
       |
       v
Positive / Negative
```

## Important notes

- The app trains the model when the user clicks **Train LSTM model**; it does not train automatically at startup.
- The model is held in Streamlit's in-memory resource cache. It is not permanently saved. A server restart or cache eviction can require retraining.
- The IMDB vocabulary is limited to the 10,000 most frequent words. Unknown words are mapped to the unknown token.
- This is an educational demonstration, not a production-grade moderation or decision system.
