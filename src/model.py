from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential

from src.config import EMBEDDING_DIM, RNN_UNITS, VOCAB_SIZE


def build_lstm_model(max_len: int, vocab_size: int = VOCAB_SIZE) -> Sequential:
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len),
            LSTM(units=RNN_UNITS),
            Dense(units=vocab_size, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
