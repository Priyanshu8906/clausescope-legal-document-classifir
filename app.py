"""DomainDesk: twelve-topic document classification for an NLP class project."""
from __future__ import annotations

import io
import re
import sys
import types
from pathlib import Path

LOCAL_PACKAGES = Path(__file__).resolve().parent / ".runtime"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

try:
    import numpy as np
    try:
        import numpy.random  # noqa: F401
    except (ImportError, OSError):
        np.random = types.ModuleType("numpy.random")
        np.random.Generator = object
        np.random.BitGenerator = object
        np.random.RandomState = object
        sys.modules["numpy.random"] = np.random
except ImportError:
    pass

import streamlit as st
from classifier_runtime import DomainClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "artifacts" / "domain-distilbert" / "model"
MODEL_FILE = MODEL_DIR / "classifier.pt"
MAX_INPUT_CHARS = 180_000

DOMAINS = {
    "legal": "Legal",
    "healthcare": "Healthcare",
    "finance": "Finance & investment",
    "education": "Education & careers",
    "business": "Business & industry",
    "manufacturing": "Industrial & manufacturing",
    "technology": "Software & technology",
    "science": "Science & research",
    "news_politics": "News & politics",
    "sports": "Sports & athletics",
    "automotive": "Autos & vehicles",
    "arts_entertainment": "Arts & entertainment",
}

EXAMPLES = {
    "Legal": "Under section 4.2, the supplier shall indemnify the customer against third-party claims arising from a breach of this agreement. Either party may terminate for material breach after 30 days' written notice. This agreement is governed by the laws of the State of New York.",
    "Healthcare": "Assessment: The patient reports persistent cough and shortness of breath. Oxygen saturation is 93% on room air. The care plan includes chest imaging, monitoring of respiratory status, and follow-up after treatment.",
    "Finance & investment": "The company reported quarterly revenue of $4.2 billion, up 8% year over year. Operating margin was 14.6%, while free cash flow reached $620 million. Management maintained its full-year earnings guidance.",
    "Education & careers": "Learning objectives: students will compare primary sources, support a written claim with evidence, and revise their work using peer feedback. The assessment rubric measures reasoning, organization, and use of evidence.",
    "Business & industry": "The company expanded its regional distribution network and hired 120 employees this quarter. Management expects the new facility to reduce delivery times and improve operating efficiency across the supply chain.",
    "Industrial & manufacturing": "Production quality report: 2.1% of units failed final inspection, primarily due to inconsistent torque at assembly station three. The corrective action is to recalibrate the tooling and add an in-process verification check.",
    "Software & technology": "The release adds an API endpoint for batch processing, improves query latency, and fixes a memory leak in the worker service. Administrators can configure authentication scopes in the deployment settings.",
    "Science & research": "The study measured changes in surface temperature across three experimental conditions. Results indicate a statistically significant difference between the control and treatment groups, although further replication is needed.",
    "News & politics": "The committee voted Tuesday to advance the proposed measure after a three-hour hearing. Lawmakers said the bill will receive a full chamber debate next month, while opposition members called for amendments.",
    "Sports & athletics": "The home side scored twice in the final quarter to secure a 3–1 win. The coach praised the midfield pressing and said the team will focus on recovery before Saturday's away fixture.",
    "Autos & vehicles": "The 2026 model adds a more efficient hybrid powertrain, updated driver-assistance features, and a redesigned interior. The manufacturer says deliveries will begin at dealerships in early spring.",
    "Arts & entertainment": "The director's new production opens with a spare piano arrangement before the ensemble enters. Critics praised the performances and lighting design, while noting that the second act moves at a slower pace.",
}

st.set_page_config(
    page_title="DomainDesk | Document classifier",
    page_icon=":material/folder_open:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the local classifier…")
def load_classifier(model_path: str):
    return DomainClassifier(model_path)


def predict_document(text: str) -> tuple[list[tuple[str, float]], int]:
    return load_classifier(str(MODEL_DIR.resolve())).predict(text)


def read_document(upload) -> str:
    data = upload.getvalue()
    suffix = Path(upload.name).suffix.lower()
    if suffix == ".txt":
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document

        document = Document(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise ValueError("Choose a .txt, .pdf, or .docx file.")


def short_preview(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text.strip()))
    preview = " ".join(sentence for sentence in sentences[:2] if sentence)
    return preview[:420] + ("…" if len(preview) > 420 else "")


def domain_name(label: str) -> str:
    return DOMAINS.get(label, label.replace("_", " ").title())


with st.sidebar:
    st.title("DomainDesk", icon=":material/folder_open:")
    st.caption("A document-domain classifier")
    st.markdown("#### Supported domains")
    st.write("Legal · Healthcare · Finance · Education")
    st.write("Business · Manufacturing · Technology · Science")
    st.write("News · Sports · Automotive · Arts & entertainment")
    st.markdown("#### How it works")
    st.caption("A locally fine-tuned DistilBERT model reads the full document in overlapping text windows and compares its score across twelve supported topics.")
    st.caption("Documents stay on this computer. Scanned PDFs need OCR before upload.")

st.title("Find the domain of a document", icon=":material/description:")
st.markdown("Drop in a document or paste text. DomainDesk reviews the entire text and shows its strongest domain matches.")

if not MODEL_FILE.exists():
    st.warning(
        "The expanded twelve-domain model is not installed yet. From the project folder, run `python download_data.py`, `python prepare_data.py`, and `python train.py`.",
        title="Model files are missing",
    )

input_mode = st.segmented_control(
    "Choose an input method",
    ["Paste text", "Upload a file"],
    default="Paste text",
    key="input_mode",
    selection_mode="single",
)

document_text = ""
if input_mode == "Paste text":
    example = st.selectbox("Try an example", ["Choose a sample…", *EXAMPLES.keys()], key="example_domain")
    initial_text = EXAMPLES.get(example, "")
    document_text = st.text_area(
        "Document text",
        key="document_text",
        value=initial_text,
        height=250,
        placeholder="Paste a contract, care note, financial report, lesson plan, business page, software document, research summary, news article, sports report, vehicle description, or arts review…",
        help="For a more reliable result, provide several sentences or a full document section.",
    )
else:
    uploaded = st.file_uploader(
        "Choose a document",
        key="document_upload",
        type=["txt", "pdf", "docx"],
        help="Selectable-text PDFs work best. Scanned pages need OCR first.",
    )
    if uploaded:
        try:
            document_text = read_document(uploaded)
            st.caption(f"Extracted {len(document_text):,} characters from {uploaded.name}.")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")

document_text = document_text.strip()
if document_text and len(document_text) > MAX_INPUT_CHARS:
    st.warning(f"This file has {len(document_text):,} characters. The app will classify the first {MAX_INPUT_CHARS:,} characters.")
    document_text = document_text[:MAX_INPUT_CHARS]

if document_text:
    words = len(document_text.split())
    if words < 35:
        st.caption("Short input can be ambiguous. A paragraph or a full document section usually gives the model more useful evidence.")
    else:
        st.caption(f"{words:,} words ready for classification")

classify = st.button(
    "Classify document",
    type="primary",
    icon=":material/search:",
    disabled=not bool(document_text) or not MODEL_FILE.exists(),
    width="stretch",
)

if classify:
    if len(document_text.split()) < 12:
        st.warning("Please provide at least a few complete sentences for a useful classification.", title="More text is needed")
    else:
        try:
            with st.spinner("Reading document sections…"):
                predictions, chunk_count = predict_document(document_text)
            best_label, best_score = predictions[0]
            heading, icon = ("Likely domain", ":material/check_circle:") if best_score >= 0.60 else ("Possible match · review alternatives", ":material/info:")
            with st.container(border=True):
                st.subheader(heading, icon=icon)
                left, right = st.columns([1, 2], vertical_alignment="center")
                with left:
                    st.metric("Top match", domain_name(best_label))
                    st.metric("Model score", f"{best_score:.1%}")
                with right:
                    st.markdown("**Other domain matches**")
                    for label, score in predictions[1:4]:
                        st.progress(score, text=f"{domain_name(label)} · {score:.1%}")
                st.caption(f"Compared {chunk_count} overlapping text sections. Scores are model signals, not calibrated probabilities; mixed or unsupported documents can be misclassified.")
            with st.expander("Document preview"):
                st.write(short_preview(document_text))
        except Exception as exc:
            st.error(f"Classification failed: {exc}")

if not classify:
    st.space("small")
    with st.container(border=True):
        st.subheader("Twelve topics, one simple workflow", icon=":material/auto_awesome:")
        st.write("The model compares your text with examples spanning legal, healthcare, finance, education, business, manufacturing, technology, science, news, sports, automotive, and arts content.")

st.caption("Educational NLP project · Not professional legal, medical, or financial advice")
