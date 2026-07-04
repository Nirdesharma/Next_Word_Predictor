import pickle
from typing import List, Tuple

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import LSTM_MODEL_PATH, MAX_LEN_PATH, TOKENIZER_PATH


def _build_index_to_word(word_index: dict) -> dict:
    return {index: word for word, index in word_index.items()}


def load_artifacts(
    model_path=LSTM_MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    max_len_path=MAX_LEN_PATH,
):
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)
    with open(max_len_path, "rb") as file:
        max_len = pickle.load(file)
    index_to_word = _build_index_to_word(tokenizer.word_index)
    return model, tokenizer, max_len, index_to_word


def preprocess_text(text: str, tokenizer, max_len: int) -> np.ndarray:
    normalized = text.lower().strip()
    sequence = tokenizer.texts_to_sequences([normalized])[0]
    if not sequence:
        raise ValueError("No known words in input. Try common English words from the quote dataset.")
    return pad_sequences([sequence], maxlen=max_len, padding="pre")


def predict_next_word(
    model,
    tokenizer,
    max_len: int,
    index_to_word: dict,
    text: str,
    top_k: int = 5,
) -> Tuple[str, List[Tuple[str, float]]]:
    padded = preprocess_text(text, tokenizer, max_len)
    probabilities = model.predict(padded, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    predictions = []
    for index in top_indices:
        word = index_to_word.get(int(index), "")
        if word:
            predictions.append((word, float(probabilities[index])))

    if not predictions:
        raise ValueError("Model could not produce a valid next-word prediction.")

    return predictions[0][0], predictions


def generate_text(
    model,
    tokenizer,
    max_len: int,
    index_to_word: dict,
    seed_text: str,
    num_words: int,
) -> str:
    text = seed_text.strip()
    for _ in range(num_words):
        next_word, _ = predict_next_word(model, tokenizer, max_len, index_to_word, text, top_k=1)
        text = f"{text} {next_word}"
    return text
