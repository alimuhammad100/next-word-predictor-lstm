import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="NextGen AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    with open("max_len.pkl", "rb") as file:
        max_len = pickle.load(file)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()


def generate_text(seed_text, next_words):
    generated_text = seed_text

    for _ in range(next_words):
        input_sequence = tokenizer.texts_to_sequences(
            [generated_text]
        )[0]

        if not input_sequence:
            break

        input_sequence = pad_sequences(
            [input_sequence],
            maxlen=max_len - 1,
            padding="pre"
        )

        prediction = model.predict(
            input_sequence,
            verbose=0
        )

        predicted_index = np.argmax(prediction)

        predicted_word = None

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break

        if predicted_word is None:
            break

        generated_text += " " + predicted_word

    return generated_text


css = (
    "<style>"

    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');"

    "* {"
    "font-family: 'Inter', sans-serif;"
    "}"

    ".stApp {"
    "background: radial-gradient(circle at 15% 20%, rgba(83,75,255,0.16), transparent 25%),"
    "radial-gradient(circle at 85% 80%, rgba(0,229,255,0.10), transparent 25%),"
    "#070B14;"
    "color: white;"
    "}"

    "header {"
    "background: transparent !important;"
    "}"

    ".block-container {"
    "max-width: 1250px;"
    "padding-top: 2rem;"
    "padding-bottom: 2rem;"
    "}"

    ".hero {"
    "text-align: center;"
    "padding: 45px 20px 35px 20px;"
    "}"

    ".badge {"
    "display: inline-block;"
    "padding: 9px 18px;"
    "border-radius: 50px;"
    "font-size: 13px;"
    "font-weight: 600;"
    "color: #B8B5FF;"
    "background: rgba(116,104,255,0.12);"
    "border: 1px solid rgba(130,120,255,0.25);"
    "margin-bottom: 22px;"
    "}"

    ".hero h1 {"
    "font-family: 'Space Grotesk', sans-serif;"
    "font-size: 58px;"
    "font-weight: 800;"
    "margin-bottom: 15px;"
    "background: linear-gradient(90deg, #FFFFFF, #9EA7FF, #66E3FF);"
    "-webkit-background-clip: text;"
    "-webkit-text-fill-color: transparent;"
    "}"

    ".hero p {"
    "color: #9BA6B8;"
    "font-size: 18px;"
    "max-width: 680px;"
    "margin: auto;"
    "line-height: 1.7;"
    "}"

    ".ai-card {"
    "background: rgba(17,24,39,0.72);"
    "border: 1px solid rgba(255,255,255,0.08);"
    "border-radius: 24px;"
    "padding: 28px;"
    "box-shadow: 0 20px 60px rgba(0,0,0,0.25);"
    "margin-top: 20px;"
    "}"

    ".section-title {"
    "font-family: 'Space Grotesk', sans-serif;"
    "font-size: 22px;"
    "font-weight: 700;"
    "margin-bottom: 6px;"
    "color: white;"
    "}"

    ".section-subtitle {"
    "color: #7F8A9D;"
    "font-size: 14px;"
    "margin-bottom: 20px;"
    "}"

    ".prediction-card {"
    "background: linear-gradient(135deg, rgba(91,78,255,0.16), rgba(0,214,255,0.08));"
    "border: 1px solid rgba(135,127,255,0.25);"
    "border-radius: 22px;"
    "padding: 35px;"
    "text-align: center;"
    "margin-top: 30px;"
    "}"

    ".prediction-label {"
    "font-size: 13px;"
    "color: #9EA7FF;"
    "text-transform: uppercase;"
    "letter-spacing: 2px;"
    "margin-bottom: 18px;"
    "}"

    ".prediction-text {"
    "font-family: 'Space Grotesk', sans-serif;"
    "font-size: 28px;"
    "font-weight: 600;"
    "color: white;"
    "line-height: 1.7;"
    "word-break: break-word;"
    "}"

    ".stat-card {"
    "background: rgba(255,255,255,0.035);"
    "border: 1px solid rgba(255,255,255,0.07);"
    "border-radius: 18px;"
    "padding: 20px;"
    "text-align: center;"
    "margin-bottom: 15px;"
    "}"

    ".stat-number {"
    "font-family: 'Space Grotesk', sans-serif;"
    "font-size: 25px;"
    "font-weight: 700;"
    "color: #A7AEFF;"
    "}"

    ".stat-label {"
    "font-size: 12px;"
    "color: #7F8A9D;"
    "margin-top: 5px;"
    "}"

    ".feature-card {"
    "background: rgba(255,255,255,0.025);"
    "border: 1px solid rgba(255,255,255,0.06);"
    "border-radius: 18px;"
    "padding: 22px;"
    "height: 100%;"
    "}"

    ".feature-icon {"
    "font-size: 28px;"
    "margin-bottom: 12px;"
    "}"

    ".feature-title {"
    "font-size: 17px;"
    "font-weight: 600;"
    "color: white;"
    "margin-bottom: 8px;"
    "}"

    ".feature-text {"
    "font-size: 13px;"
    "color: #8691A5;"
    "line-height: 1.6;"
    "}"

    ".footer {"
    "text-align: center;"
    "color: #697386;"
    "font-size: 13px;"
    "padding: 50px 0 20px 0;"
    "}"

    ".footer span {"
    "color: #9EA7FF;"
    "font-weight: 600;"
    "}"

    "div.stButton > button {"
    "width: 100%;"
    "height: 55px;"
    "border: none;"
    "border-radius: 14px;"
    "font-size: 16px;"
    "font-weight: 600;"
    "color: white;"
    "background: linear-gradient(90deg, #6257FF, #3B82F6);"
    "}"

    "div.stButton > button:hover {"
    "transform: translateY(-2px);"
    "box-shadow: 0 10px 30px rgba(91,78,255,0.35);"
    "}"

    "div[data-testid='stTextArea'] textarea,"
    "div[data-baseweb='textarea'] textarea {"
    "background: #FFFFFF !important;"
    "background-color: #FFFFFF !important;"
    "color: #000000 !important;"
    "-webkit-text-fill-color: #000000 !important;"
    "caret-color: #000000 !important;"
    "border-radius: 14px !important;"
    "border: 1px solid #D1D5DB !important;"
    "font-size: 16px !important;"
    "}"

    "div[data-testid='stTextArea'] textarea:focus,"
    "div[data-baseweb='textarea'] textarea:focus {"
    "background: #FFFFFF !important;"
    "background-color: #FFFFFF !important;"
    "color: #000000 !important;"
    "-webkit-text-fill-color: #000000 !important;"
    "border: 1px solid #6257FF !important;"
    "box-shadow: 0 0 0 1px #6257FF !important;"
    "}"

    "div[data-testid='stTextArea'] textarea::placeholder,"
    "div[data-baseweb='textarea'] textarea::placeholder {"
    "color: #6B7280 !important;"
    "-webkit-text-fill-color: #6B7280 !important;"
    "opacity: 1 !important;"
    "}"

    "div[data-testid='stTextArea'] label {"
    "color: white !important;"
    "}"

    "#MainMenu {"
    "visibility: hidden;"
    "}"

    "footer {"
    "visibility: hidden;"
    "}"

    "</style>"
)

st.markdown(
    css,
    unsafe_allow_html=True
)


hero = (
    "<div class='hero'>"

    "<div class='badge'>"
    "✦ POWERED BY DEEP LEARNING"
    "</div>"

    "<h1>Continue Your Thoughts with AI.</h1>"

    "<p>"
    "Start writing and let artificial intelligence intelligently "
    "continue your sentence using deep learning."
    "</p>"

    "</div>"
)

st.markdown(
    hero,
    unsafe_allow_html=True
)


left, right = st.columns([1.6, 1])


with left:

    st.markdown(
        "<div class='ai-card'>"
        "<div class='section-title'>✍️ Start Writing</div>"
        "<div class='section-subtitle'>"
        "Enter a sentence and let the AI intelligently continue it."
        "</div>",
        unsafe_allow_html=True
    )

    user_input = st.text_area(
        "Your text",
        placeholder="Start writing something... For example: Artificial intelligence is changing the",
        height=180,
        label_visibility="collapsed"
    )

    words_to_generate = st.slider(
        "Words to generate",
        min_value=1,
        max_value=30,
        value=10
    )

    predict_button = st.button(
        "✨ Generate with AI"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


with right:

    st.markdown(
        "<div class='ai-card'>"

        "<div class='section-title'>🧠 AI Engine</div>"

        "<div class='section-subtitle'>"
        "Advanced language generation powered by deep learning."
        "</div>"

        "<div class='stat-card'>"
        "<div class='stat-number'>LSTM</div>"
        "<div class='stat-label'>Neural Architecture</div>"
        "</div>"

        "<div class='stat-card'>"
        "<div class='stat-number'>NLP</div>"
        "<div class='stat-label'>Language Processing</div>"
        "</div>"

        "<div class='stat-card'>"
        "<div class='stat-number'>AI</div>"
        "<div class='stat-label'>Text Generation</div>"
        "</div>"

        "</div>",
        unsafe_allow_html=True
    )


if predict_button:

    if user_input.strip() == "":

        st.warning(
            "Please enter a sentence before generating text."
        )

    else:

        with st.spinner(
            "🧠 AI is analyzing your sentence and generating text..."
        ):

            generated_text = generate_text(
                user_input,
                words_to_generate
            )

        prediction_html = (
            "<div class='prediction-card'>"

            "<div class='prediction-label'>"
            "AI Generated Text"
            "</div>"

            "<div class='prediction-text'>"
            + generated_text +
            "</div>"

            "</div>"
        )

        st.markdown(
            prediction_html,
            unsafe_allow_html=True
        )


st.markdown(
    "<br><br>",
    unsafe_allow_html=True
)


st.markdown(
    "<div style='text-align:center; margin-bottom:25px;'>"

    "<div class='section-title'>"
    "Built for Intelligent Writing"
    "</div>"

    "<div class='section-subtitle'>"
    "Advanced deep learning combined with a premium AI experience."
    "</div>"

    "</div>",
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        "<div class='feature-card'>"

        "<div class='feature-icon'>⚡</div>"

        "<div class='feature-title'>"
        "Instant Generation"
        "</div>"

        "<div class='feature-text'>"
        "Generate multiple words and intelligently continue your thoughts."
        "</div>"

        "</div>",
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        "<div class='feature-card'>"

        "<div class='feature-icon'>🧠</div>"

        "<div class='feature-title'>"
        "Deep Learning"
        "</div>"

        "<div class='feature-text'>"
        "Powered by an LSTM neural network trained to understand language patterns."
        "</div>"

        "</div>",
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        "<div class='feature-card'>"

        "<div class='feature-icon'>✨</div>"

        "<div class='feature-title'>"
        "Smart Context"
        "</div>"

        "<div class='feature-text'>"
        "The AI uses your sentence context to generate meaningful continuations."
        "</div>"

        "</div>",
        unsafe_allow_html=True
    )


st.markdown(
    "<div class='footer'>"

    "Designed & Developed by <span>Ali Muhammad</span>"

    "<br>"

    "NextGen AI • Intelligent Text Generation"

    "<br><br>"

    "© 2026 Ali Muhammad • All Rights Reserved"

    "</div>",
    unsafe_allow_html=True
)