# Next Word Predictor

An end-to-end **Natural Language Processing (NLP)** project that predicts the next word in a sentence using a **Long Short-Term Memory (LSTM)** neural network. The model is trained on **3,038 inspirational quotes** and served through an interactive **Streamlit** web application.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## Highlights (for portfolio / resume)

- Built a complete NLP pipeline: text cleaning → tokenization → sequence modeling → inference
- Designed an **Embedding + LSTM + Softmax** architecture for multi-class next-word prediction
- Trained on **85,000+ prefix-target pairs** derived from quote data
- Deployed a polished **Streamlit demo** with next-word ranking and text generation
- Modular codebase with separate training, inference, and UI layers


## Demo features

| Page | Description |
|------|-------------|
| **Home** | Project overview and training metrics |
| **Predict Next Word** | Top-k next word suggestions with confidence bars |
| **Generate Text** | Greedy multi-word continuation from a seed phrase |
| **Project Details** | Architecture, folder structure, resume text |

## Quick start

### 1. Clone and install

```bash
cd Next_word_predictor
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Train the model

Quick demo (few minutes):

```bash
python -m src.train --epochs 5
```

Full notebook replication (~100 epochs):

```bash
python -m src.train --epochs 100
```

Artifacts are saved to `models/`:
- `lstm_model.keras`
- `tokenizer.pkl`
- `max_len.pkl`
- `training_metadata.json`

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Project structure

```
Next_word_predictor/
├── app.py                      # Streamlit frontend
├── src/
│   ├── config.py               # Paths and hyperparameters
│   ├── data_pipeline.py        # Data loading & preprocessing
│   ├── model.py                # LSTM architecture
│   ├── predict.py              # Inference & text generation
│   └── train.py                # Training script
├── models/                     # Generated model artifacts
├── qoute_dataset.csv           # Quote dataset (3,038 rows)
├── Next_Word_Prediction.ipynb  # Original notebook & EDA
├── requirements.txt
└── README.md
```

## Model architecture

```
Input Sequence (padded)
        ↓
Embedding (vocab=8978, dim=50)
        ↓
LSTM (128 units)
        ↓
Dense Softmax (8978 classes)
        ↓
Next-word probability distribution
```

## Dataset

- **Source:** Inspirational quotes (`qoute_dataset.csv`)
- **Size:** 3,038 quotes, 8,978 vocabulary tokens
- **Preprocessing:** Lowercasing, punctuation removal, Keras Tokenizer

## Tech stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras (LSTM, Embedding)
- **Data:** Pandas, NumPy
- **Frontend:** Streamlit
- **Experimentation:** Jupyter Notebook

## Example usage (Python)

```python
from src.predict import load_artifacts, predict_next_word

model, tokenizer, max_len, index_to_word = load_artifacts()
word, top_k = predict_next_word(model, tokenizer, max_len, index_to_word, "what are you")
print(word)       # e.g. implying
print(top_k[:3])  # top 3 suggestions with scores
```


