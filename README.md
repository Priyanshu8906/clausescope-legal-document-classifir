# DomainDesk — document topic classifier

DomainDesk predicts the topic of English document text across **12 categories**: legal, healthcare, finance, education, business, manufacturing, technology, science, news and politics, sports, automotive, and arts and entertainment. It fine-tunes DistilBERT and provides a polished Streamlit interface for pasted text or `.txt`, `.pdf`, and `.docx` uploads.

> Educational NLP project. Predictions are best-effort topic matches, not legal, medical, or financial advice. The model always selects among its supported categories, and its score is not a calibrated probability.

## Features

- Fine-tuned 12-class DistilBERT model with held-out evaluation.
- English document-topic examples from an attributed web-document corpus, plus focused LEDGAR, PubMedQA, and Financial PhraseBank data.
- Balanced sampling during training to account for the supplemental class examples.
- Whole-document inference using overlapping text windows instead of silently classifying only the opening paragraph.
- Streamlit UI with a calm green theme, sample inputs, upload or paste modes, top alternatives, and a brief text preview.
- Local inference after dependencies and model files are installed; uploaded text is not sent to a hosted classifier.

The broad-topic dataset uses upstream labels created by an annotation pipeline. Some labels may be noisy or ambiguous. The focused legal, healthcare, and finance corpora add examples from those domains. See [`DATASETS.md`](DATASETS.md) for sources, license notes, and the scope of the prepared experiment.

## Open and run in VS Code

Clone the private repository or open its folder in VS Code. In the integrated PowerShell terminal:

```powershell
git lfs pull
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_app.py --global.developmentMode false --server.port 8501 --server.address 127.0.0.1
```

Open the local URL printed in the terminal. If port 8501 is already occupied, replace it with `8502` in the run command and open `http://localhost:8502`. Keep the terminal open while using the app.

## Rebuild the training data and model

The tracked repository includes the prepared splits and fine-tuned model. To reproduce data preparation from public sources, the downloader fetches selected topic categories (about 500 MB total) and the focused domain datasets:

```powershell
python download_data.py
python prepare_data.py
python train.py --epochs 2 --max-length 192 --batch-size 8
```

The prepared set contains **2,400 training**, **540 validation**, and **540 test** examples. The web-document source provides disjoint train and test partitions; validation examples are reserved from the training side. LEDGAR, PubMedQA, and Financial PhraseBank add disjoint supplemental examples to legal, healthcare, and finance. The training loader uses class-balanced sampling; `data/processed/metadata.json` records the exact counts for each class.

The first fine-tuning run downloads the DistilBERT base checkpoint. CPU training works but takes longer than using a supported CUDA GPU. The training script saves the model, `metrics.json`, `label_map.json`, and a confusion matrix under `artifacts/domain-distilbert/`.

## Limitations

- Labels come from curated web-document topics and focused source datasets; some examples may be ambiguous, and the model can learn source or writing-style cues.
- The supported labels are English-language topics, not every possible industry or document type.
- The model returns a supported category even for out-of-scope text. Compare the top alternatives for mixed-domain documents.
- Scanned PDFs need OCR. Long files are processed in overlapping windows up to the app's character limit.
- The displayed softmax score is a model score, not a calibrated probability.

## Project files

- `app.py` — Streamlit interface and document extraction/inference.
- `download_data.py` — public source downloader.
- `prepare_data.py` — cleaning, sampling, and split creation.
- `train.py` — DistilBERT fine-tuning and evaluation.
- `DATASETS.md` — data attribution and license notes.
- `PROJECT_REPORT.md` — methodology, test results, and limitations.
