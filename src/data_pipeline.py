import pickle
import string

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from src.config import DATA_PATH, MAX_LEN_PATH, TOKENIZER_PATH, VOCAB_SIZE


def load_quotes(csv_path=DATA_PATH) -> pd.Series:
    df = pd.read_csv(csv_path)
    return df["quote"]


def clean_quotes(quotes: pd.Series) -> pd.Series:
    translator = str.maketrans("", "", string.punctuation)
    return quotes.apply(lambda x: x.translate(translator)).str.lower()


def build_training_data(quotes: pd.Series, vocab_size: int = VOCAB_SIZE):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(quotes)

    sequences = tokenizer.texts_to_sequences(quotes)

    x_samples, y_samples = [], []
    for seq in sequences:
        for i in range(1, len(seq)):
            x_samples.append(seq[:i])
            y_samples.append(seq[i])

    max_len = max(len(x) for x in x_samples)
    x_padded = pad_sequences(x_samples, maxlen=max_len, padding="pre")
    y_array = np.array(y_samples, dtype=np.int32)

    return tokenizer, x_padded, y_array, max_len


def save_preprocessing_artifacts(tokenizer, max_len, tokenizer_path=TOKENIZER_PATH, max_len_path=MAX_LEN_PATH):
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tokenizer_path, "wb") as file:
        pickle.dump(tokenizer, file)
    with open(max_len_path, "wb") as file:
        pickle.dump(max_len, file)


def load_preprocessing_artifacts(tokenizer_path=TOKENIZER_PATH, max_len_path=MAX_LEN_PATH):
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)
    with open(max_len_path, "rb") as file:
        max_len = pickle.load(file)
    return tokenizer, max_len
