import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("bert_bias_model")
    tokenizer = AutoTokenizer.from_pretrained("bert_bias_model")
    return tokenizer, model

tokenizer, model = load_model()
model.eval()

label_map = {
    0: -1,  # Left
    1: 0,   # Neutral
    2: 1    # Right
}

# Streamlit UI
st.title("🗞️ News Bias Detection")
st.write("Enter a news article snippet to detect its political bias.")

user_input = st.text_area("📝 Input Text", height=200)

if st.button("Analyze Bias"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize and predict
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        bias_label = label_map[pred]
        label_name = { -1: "Left", 0: "Neutral", 1: "Right" }[bias_label]
        
        st.markdown(f"### 🔍 Prediction: **{label_name}**")
        st.markdown(f"Confidence: `{confidence:.2%}`")
