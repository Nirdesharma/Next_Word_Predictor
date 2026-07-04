from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "qoute_dataset.csv"
MODELS_DIR = ROOT_DIR / "models"

LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
MAX_LEN_PATH = MODELS_DIR / "max_len.pkl"
METADATA_PATH = MODELS_DIR / "training_metadata.json"

VOCAB_SIZE = 8978
EMBEDDING_DIM = 50
RNN_UNITS = 128
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
