import streamlit as st
from app.model_loader import load_model
from app.sentiment import predict_sentiment

# Load model, tokenizer, device once (cached)
@st.cache_resource
def load_resources():
    tokenizer, model, device = load_model()
    return tokenizer, model, device

tokenizer, model, device = load_resources()

st.set_page_config(
    page_title="Nepali Sentiment Analyzer",
    page_icon="🧠",
)

st.title("🧠 Nepali Sentiment Analyzer")
st.write("Type a Nepali sentence and get its sentiment: **negative**, **neutral**, or **positive**.")

text = st.text_area("Enter Nepali text here:", height=150)

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        result = predict_sentiment(text, tokenizer, model, device)
        label = result["label"]

        # Show only label nicely
        if label.lower() == "positive":
            st.success("Sentiment: **Positive** ")
        elif label.lower() == "negative":
            st.error("Sentiment: **Negative** ")
        else:
            st.info("Sentiment: **Neutral** ")
