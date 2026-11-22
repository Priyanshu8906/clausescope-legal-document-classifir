"""ClauseScope: local educational legal clause classification demo."""
from __future__ import annotations

import re
import sys
import types
from functools import lru_cache
from pathlib import Path

LOCAL_PACKAGES = Path(__file__).resolve().parent / ".runtime"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

try:
    import numpy as np
    try:
        import numpy.random  # noqa: F401
    except (ImportError, OSError):
        # The host's Windows application-control policy blocks this optional
        # extension; the UI does not use NumPy random-number functions.
        np.random = types.ModuleType("numpy.random")
        np.random.Generator = object
        np.random.BitGenerator = object
        np.random.RandomState = object
        sys.modules["numpy.random"] = np.random
except ImportError:
    pass

import streamlit as st

ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "ledgar-distilbert" / "model"
MAX_CHARS = 60_000

st.set_page_config(page_title="ClauseScope | Legal NLP", page_icon="⚖️", layout="wide")

@lru_cache(maxsize=1)
def load_classifier(model_path: str):
    from transformers import pipeline
    return pipeline("text-classification", model=model_path, tokenizer=model_path,
                    top_k=None, truncation=True, max_length=256)

def read_document(upload) -> str:
    data = upload.getvalue()
    suffix = Path(upload.name).suffix.lower()
    if suffix == ".txt":
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document
        import io
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Choose a .txt, .pdf, or .docx file.")

def summarize(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text.strip()))
    summary = " ".join(s for s in sentences[:2] if s)
    return summary[:500] + ("…" if len(summary) > 500 else "")

st.markdown("# ⚖️ ClauseScope")
st.caption("Legal clause classification · NLP class project · DistilBERT fine-tuned on LEDGAR")
st.info("Educational prototype only. This model labels short contract provisions; it is not legal advice and does not assess enforceability.")

if not ARTIFACT_DIR.exists():
    st.warning("No trained model is installed yet. From the project folder, run `python train.py --top-k 8 --epochs 2`, then reload this page.")

left, right = st.columns([1.15, .85], gap="large")
with left:
    st.subheader("Provide a provision")
    uploaded = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"], help="Scanned PDFs without selectable text need OCR first.")
    pasted = st.text_area("Or paste text", height=260, placeholder="Paste one legal clause or provision here…")
    chosen = pasted.strip()
    if uploaded:
        try:
            chosen = read_document(uploaded).strip()
            st.caption(f"Extracted {len(chosen):,} characters from {uploaded.name}.")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")
    if chosen and len(chosen) > MAX_CHARS:
        st.warning(f"Input is long ({len(chosen):,} characters); the app will use the first {MAX_CHARS:,} characters before model tokenization.")
    classify = st.button("Classify provision", type="primary", disabled=not bool(chosen) or not ARTIFACT_DIR.exists(), use_container_width=True)

with right:
    st.subheader("Prediction")
    if classify:
        try:
            with st.spinner("Running the local model…"):
                results = load_classifier(str(ARTIFACT_DIR.resolve()))(chosen[:MAX_CHARS])[0]
            results = sorted(results, key=lambda item: item["score"], reverse=True)
            best = results[0]
            st.markdown(f"### {best['label'].replace('_', ' ').title()}")
            st.metric("Model confidence", f"{best['score']:.1%}")
            st.progress(float(best["score"]))
            st.caption("Softmax confidence is not a calibrated probability. A closed-set model always selects from its known labels.")
            st.markdown("**Brief extractive summary**")
            st.write(summarize(chosen))
            st.markdown("**Other likely labels**")
            for item in results[1:4]:
                st.write(f"{item['label'].replace('_', ' ').title()} · {item['score']:.1%}")
            if len(chosen.split()) > 220:
                st.warning("This is much longer than the short provisions used for training. Only the beginning may influence the prediction.")
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
    else:
        st.write("Your category, confidence, and brief text summary will appear here.")

with st.expander("About the model and data"):
    st.write("LEDGAR is a public dataset of labeled clauses from U.S. contracts. The project selects the most frequent labels from the training split, fine-tunes DistilBERT, and evaluates on the dataset's held-out test split. Results depend on the labels and run settings. Training and inference run locally after model files are downloaded.")
