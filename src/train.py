import argparse
import json
from datetime import datetime, timezone

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    LSTM_MODEL_PATH,
    METADATA_PATH,
    MODELS_DIR,
    VALIDATION_SPLIT,
)
from src.data_pipeline import build_training_data, clean_quotes, load_quotes, save_preprocessing_artifacts
from src.model import build_lstm_model


def train(epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    quotes = clean_quotes(load_quotes())
    tokenizer, x_padded, y_labels, max_len = build_training_data(quotes)

    model = build_lstm_model(max_len)
    history = model.fit(
        x_padded,
        y_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(LSTM_MODEL_PATH)
    save_preprocessing_artifacts(tokenizer, max_len)

    final_epoch = history.history
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_split": VALIDATION_SPLIT,
        "num_quotes": int(len(quotes)),
        "num_training_samples": int(len(x_padded)),
        "max_sequence_length": int(max_len),
        "final_train_accuracy": float(final_epoch["accuracy"][-1]),
        "final_val_accuracy": float(final_epoch["val_accuracy"][-1]),
        "final_train_loss": float(final_epoch["loss"][-1]),
        "final_val_loss": float(final_epoch["val_loss"][-1]),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train the LSTM next-word prediction model.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    metadata = train(epochs=args.epochs, batch_size=args.batch_size)
    print("\nTraining complete.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
