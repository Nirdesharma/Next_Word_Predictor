import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lstm_model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"

st.set_page_config(
    page_title="Next Word Predictor | LSTM NLP",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.25rem; }
    .sub-header { color: #6b7280; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .tech-pill {
        display: inline-block; background-color: #eef2ff; color: #4338ca;
        padding: 0.25rem 0.75rem; border-radius: 999px; margin: 0.15rem;
        font-size: 0.85rem; font-weight: 600;
    }
    .prediction-box {
        background-color: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1rem 1.25rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def artifacts_ready() -> bool:
    return MODEL_PATH.exists() and TOKENIZER_PATH.exists() and MAX_LEN_PATH.exists()


@st.cache_resource(show_spinner="Loading LSTM model...")
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)
    with open(MAX_LEN_PATH, "rb") as file:
        max_len = pickle.load(file)

    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_to_word


def predictor(model, tokenizer, index_to_word, text, max_len):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    if not seq:
        raise ValueError("No known words in input. Try words from the quote dataset.")
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    pred = model.predict(seq, verbose=0)
    pred_index = int(np.argmax(pred))
    return index_to_word[pred_index], pred[0]


def predict_top_k(model, tokenizer, index_to_word, text, max_len, top_k=5):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    if not seq:
        raise ValueError("No known words in input. Try words from the quote dataset.")
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    probabilities = model.predict(seq, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    results = []
    for index in top_indices:
        word = index_to_word.get(int(index), "")
        if word:
            results.append((word, float(probabilities[index])))
    return results


def generate_text(model, tokenizer, index_to_word, seed_text, max_len, n_words):
    text = seed_text
    for _ in range(n_words):
        next_word = predictor(model, tokenizer, index_to_word, text, max_len)[0]
        if next_word == "":
            break
        text += " " + next_word
    return text


def render_sidebar():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Predict Next Word", "Generate Text", "About Project"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Tech stack**")
    for label in ["Python", "TensorFlow/Keras", "LSTM", "Streamlit"]:
        st.sidebar.markdown(f'<span class="tech-pill">{label}</span>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    if artifacts_ready():
        st.sidebar.success("Model files found")
    else:
        st.sidebar.error("Model files missing")
        st.sidebar.caption("Run the notebook cells that save `lstm_model.h5`, `tokenizer.pkl`, and `max_len.pkl`.")
    return page


def render_home():
    st.markdown('<p class="main-header">Next Word Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">LSTM-based next-word prediction trained on inspirational quotes — interactive Streamlit demo.</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", "3,038 quotes")
    c2.metric("Vocabulary", "8,978 words")
    c3.metric("Model", "LSTM")
    c4.metric("UI", "Streamlit")
    st.markdown(
        """
        ### Features
        - Predict the single most likely next word from a seed phrase
        - View **top-k** suggestions with confidence scores
        - Generate multi-word continuations greedily from seed text
        """
    )


def render_predict(model, tokenizer, max_len, index_to_word):
    st.markdown('<p class="main-header">Predict Next Word</p>', unsafe_allow_html=True)

    if "predict_seed" not in st.session_state:
        st.session_state.predict_seed = "what are you"

    presets = ["what are you", "life is what", "the only way", "love is", "you must be"]
    seed_text = st.text_input("Seed text", key="predict_seed")
    top_k = st.slider("Top suggestions", 3, 10, 5)

    cols = st.columns(len(presets))
    for i, preset in enumerate(presets):
        if cols[i].button(preset, use_container_width=True, key=f"preset_{i}"):
            st.session_state.predict_seed = preset
            st.rerun()

    if st.button("Predict", type="primary"):
        if not seed_text.strip():
            st.warning("Enter some text first.")
            return
        try:
            top_word, _ = predictor(model, tokenizer, index_to_word, seed_text, max_len)
            ranked = predict_top_k(model, tokenizer, index_to_word, seed_text, max_len, top_k)
        except ValueError as error:
            st.error(str(error))
            return

        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown(f"**Input:** `{seed_text}`")
        st.markdown(f"**Top prediction:** `{top_word}`")
        st.markdown("</div>", unsafe_allow_html=True)
        st.bar_chart({word: score for word, score in ranked})


def render_generate(model, tokenizer, max_len, index_to_word):
    st.markdown('<p class="main-header">Generate Text</p>', unsafe_allow_html=True)
    seed_text = st.text_area("Seed text", value="are you a", height=80)
    n_words = st.slider("Words to generate", 1, 30, 10)

    if st.button("Generate", type="primary"):
        if not seed_text.strip():
            st.warning("Enter a seed phrase first.")
            return
        with st.spinner("Generating..."):
            try:
                output = generate_text(model, tokenizer, index_to_word, seed_text, max_len, n_words)
            except ValueError as error:
                st.error(str(error))
                return
        st.success("Done")
        st.markdown(f"**Result:** {output}")


def render_about():
    st.markdown('<p class="main-header">About Project</p>', unsafe_allow_html=True)
    st.markdown(
        """
        **Next Word Predictor** is an NLP project that uses an LSTM neural network to predict
        the next word in a sentence, trained on a dataset of inspirational quotes.

        ### Architecture
        ```
        Embedding → LSTM (128 units) → Dense Softmax
        ```
        """
    )


def main():
    page = render_sidebar()

    if not artifacts_ready():
        st.markdown('<p class="main-header">Next Word Predictor</p>', unsafe_allow_html=True)
        st.error("Model artifacts not found in the project folder.")
        st.markdown(
            """
            Train and save the model in **`Next_Word_Prediction.ipynb`**, then run:

            ```bash
            streamlit run app.py
            ```

            Expected files in the project root:
            - `lstm_model.h5`
            - `tokenizer.pkl`
            - `max_len.pkl`
            """
        )
        if page == "About Project":
            render_about()
        return

    model, tokenizer, max_len, index_to_word = load_artifacts()

    if page == "Home":
        render_home()
    elif page == "Predict Next Word":
        render_predict(model, tokenizer, max_len, index_to_word)
    elif page == "Generate Text":
        render_generate(model, tokenizer, max_len, index_to_word)
    elif page == "About Project":
        render_about()


if __name__ == "__main__":
    main()
