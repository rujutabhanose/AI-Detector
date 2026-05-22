import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL = "google-bert/bert-base-cased"
ADAPTER = "gouwsxander/slop-detector-bert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model = PeftModel.from_pretrained(base, ADAPTER).to(DEVICE)
    model.eval()
    return model, tokenizer


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n") if len(p.strip()) > 7]


def ai_probabilities(paragraphs: list[str], model, tokenizer) -> list[float]:
    inputs = tokenizer(
        paragraphs,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()


st.set_page_config(page_title="AI Slop Detector", page_icon="🤖")
st.title("AI Slop Detector")

text = st.text_area("Paste text to classify", height=240, placeholder="One or more paragraphs…")

if st.button("Classify", type="primary", disabled=not text.strip()):
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        st.warning("No paragraphs long enough to classify (need >7 chars).")
    else:
        model, tokenizer = load_model()
        probs = ai_probabilities(paragraphs, model, tokenizer)

        overall = sum(probs) / len(probs)
        verdict = "Likely AI-written" if overall >= 0.5 else "Likely human-written"
        st.metric("AI-written likelihood", f"{overall:.1%}", delta=verdict, delta_color="off")
        st.progress(overall)

        st.divider()
        st.subheader("Per paragraph")
        for para, p in zip(paragraphs, probs):
            st.progress(p, text=f"{p:.1%} AI-written")
            st.caption(para[:200] + ("…" if len(para) > 200 else ""))
