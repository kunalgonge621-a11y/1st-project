import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Contextual Sentiment Chatbot", page_icon="🧠", layout="centered")

st.title("🧠Sentiment Analysis Chatbot")
st.markdown("This app analyzes your statement fully—detecting overall sentiment and key aspects for deep understanding.")

@st.cache_resource
def load_model():
    # You can use other models such as 'cardiffnlp/twitter-roberta-base-sentiment' for finer context
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def generate_reply(sentiment, aspects):
    # Basic bot logic, optionally expand with aspect-specific replies
    base = {
        "positive": "That’s great to hear!",
        "negative": "Sorry to hear that.",
        "neutral": "Thanks for sharing."
    }
    reply = base.get(sentiment.lower(), "Thank you for your message.")
    if aspects:
        reply += f" I noticed you mentioned: {', '.join(aspects)}."
    return reply

# Optionally use spaCy or other library for aspect/theme extraction
def extract_aspects(text):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("chat_form", clear_on_submit=True):
    user_message = st.text_area("Your Message:", height=100)
    submit_button = st.form_submit_button("Send")

if submit_button and user_message.strip():
    aspects = extract_aspects(user_message)  # Key themes/aspects of statement
    results = load_model()(user_message)
    sentiment = results[0]["label"]
    score = results[0]["score"]

    bot_reply = generate_reply(sentiment, aspects)

    st.session_state.history.append({
        "user": user_message,
        "aspects": aspects,
        "bot": bot_reply,
        "sentiment": sentiment,
        "confidence": score
    })

for chat in reversed(st.session_state.history):
    st.markdown(f"**You:** {chat['user']}")
    if chat["aspects"]:
        st.markdown(f"Aspects detected: {', '.join(chat['aspects'])}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown(f"**Sentiment:** {chat['sentiment']}  |  Confidence: {chat['confidence']:.3f}")
    st.markdown("---")
