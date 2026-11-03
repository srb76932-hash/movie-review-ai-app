import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

st.set_page_config(
    page_title="AI Movie Review Assistant",
    page_icon="🎬"
)

st.title("🎬 AI Movie Review Assistant")
st.write("Explore Encoder-only and Decoder-only Language Models.")

# Your Hugging Face repositories
DISTILBERT_REPO = "shireen126/distilbert-imdb-sentiment"
GPT2_REPO = "shireen126/gpt2-imdb-generation"

@st.cache_resource
def load_distilbert():
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_REPO
    )
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_gpt2():
    tokenizer = AutoTokenizer.from_pretrained(GPT2_REPO)
    model = AutoModelForCausalLM.from_pretrained(GPT2_REPO)
    model.eval()
    return tokenizer, model

choice = st.selectbox(
    "Select a Model",
    [
        "DistilBERT - Sentiment Classification",
        "GPT-2 - Text Generation"
    ]
)

if choice.startswith("DistilBERT"):

    st.subheader("Movie Sentiment Analysis")

    review = st.text_area("Enter your movie review:")

    if st.button("Analyze Sentiment"):

        if review.strip():

            with st.spinner("Analyzing review..."):

                tokenizer, model = load_distilbert()

                inputs = tokenizer(
                    review,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                )

                with torch.no_grad():
                    outputs = model(**inputs)

                probabilities = torch.softmax(
                    outputs.logits, dim=-1
                )[0]

                prediction = torch.argmax(probabilities).item()

                label = model.config.id2label.get(
                    prediction,
                    str(prediction)
                )

                st.success(f"Prediction: {label}")
                st.write(
                    f"Confidence: {probabilities[prediction].item():.2%}"
                )

        else:
            st.warning("Please enter a movie review.")

else:

    st.subheader("Movie Review Text Generation")

    prompt = st.text_area(
        "Enter a starting sentence:",
        value="This movie was amazing because"
    )

    if st.button("Generate Text"):

        if prompt.strip():

            with st.spinner("Generating text..."):

                tokenizer, model = load_gpt2()

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt"
                )

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=60,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )

                generated_text = tokenizer.decode(
                    output[0],
                    skip_special_tokens=True
                )

                st.success("Generated Text:")
                st.write(generated_text)

        else:
            st.warning("Please enter a starting sentence.")