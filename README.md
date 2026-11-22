# ClauseScope — Legal Document Classifier

A small, reproducible NLP project that classifies **legal contract provisions** into their clause categories. It fine-tunes DistilBERT on the public **LEDGAR** subset of LexGLUE, reports held-out evaluation metrics, and provides a Streamlit interface for pasted text or uploaded documents.

> This is an educational research prototype. It does not interpret a contract, provide legal advice, or establish that a clause is legally enforceable. Predictions are only as reliable as the training data and may be wrong, especially for long or out-of-domain documents.

## What it does

- Downloads and stores the official train, validation, and test splits of LEDGAR as Parquet files.
- Selects the most frequent clause labels using **training split only**, filters rare classes, and fine-tunes `distilbert-base-uncased` as a single-label classifier.
- Reports accuracy, macro F1, weighted F1, and a confusion matrix on the dataset's held-out test split.
- Saves the model, tokenizer, label map, and evaluation report under `artifacts/ledgar-distilbert/`.
- Runs a local Streamlit UI. Input files can be `.txt`, `.pdf`, or `.docx`.

The project does not use a made-up dataset. The download script retrieves the real public Parquet splits and training reads them locally. The dataset contains short provisions, so this model classifies provisions, not arbitrary whole contracts. The app warns users when input is long and truncates to the model's 256-token limit.

## Setup

Python 3.10–3.12 is recommended. Install CPU or GPU-enabled PyTorch appropriate for your machine first if the default PyPI wheel is unsuitable, then:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The pretrained base model is downloaded at runtime, so training needs internet access. Training is compute-intensive; a CUDA GPU is recommended. The defaults cap training at 400 examples per selected category and evaluation at 250 per category so a student workstation can finish. Increase those caps (or pass 0) for a larger experiment. Report the sample sizes from `metrics.json`.

## Train and evaluate

Download and keep a local copy of all three dataset splits (Parquet):

```bash
python download_data.py
```

The training script reads these local files.

```bash
python train.py --top-k 8 --epochs 2
```

Optional controls:

```bash
python train.py --top-k 8 --epochs 2 --max-train-per-class 800 --max-eval-per-class 500
```

`--max-*` controls limit examples per category; 0 uses every available example. The script writes `model/`, `metrics.json`, `label_map.json`, and a confusion-matrix SVG under `artifacts/ledgar-distilbert/`. It prints the selected labels and sample counts. Use validation metrics for iteration and keep the test split held out for evaluation.

## Run the app

```bash
python run_app.py
```

Train the model first. The app explains how to train it if no artifact exists. It does not silently substitute a keyword or mock classifier.

## Project structure

```text
app.py                 Streamlit user interface
train.py               Dataset preparation, fine-tuning, evaluation, artifacts
requirements.txt       Python dependencies
```

## Method and limitations

LEDGAR is a legal-domain dataset of labeled provisions from U.S. contracts. This is a **closed-set** classifier: it can only return one of the selected training labels. An unfamiliar clause may still receive a confident-looking label. The displayed softmax score is model confidence, not calibrated probability. No contract data is sent to a hosted inference service by this app, but the base-model download requires internet during training.

For a class report, include the run's `metrics.json`, label distribution, confusion matrix, selected labels, training configuration, and a short error analysis. Do not report sample-run results as benchmark results. Cite the LexGLUE / LEDGAR paper and dataset card in your submission and verify the dataset's applicable license/terms before redistribution.

## References

- Chalkidis et al. (2022), *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*, ACL 2022.
- Hugging Face dataset: `lex_glue`, configuration `ledgar`.
- Base checkpoint: `distilbert/distilbert-base-uncased`.
