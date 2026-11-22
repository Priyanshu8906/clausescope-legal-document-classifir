# DomainDoc — Document Domain Classifier

DomainDoc is an NLP class project that predicts whether a text document belongs to the **legal**, **healthcare**, or **finance** domain. It fine-tunes DistilBERT on real public text datasets, evaluates on held-out data, and offers a Streamlit interface for pasted text or `.txt`, `.pdf`, and `.docx` uploads.

> This is an educational prototype. It classifies domain only; it does not provide legal, medical, or financial advice. Its confidence score is not a calibrated probability.

## What is included

- Source data for LEDGAR (legal), Financial PhraseBank (finance), and PubMedQA (healthcare).
- A reproducible preprocessing step that builds balanced labeled splits with seed 42.
- A DistilBERT fine-tuning script and held-out metrics/confusion matrix.
- A local Streamlit app for document upload or pasted text.

The source datasets have different formats and lengths: LEDGAR contains short legal provisions, Financial PhraseBank contains finance news sentences, and PubMedQA provides biomedical questions, abstract context, and answers. They are combined for a three-domain classroom prototype. The model is not a general-purpose whole-file document understanding system; it uses the first 192 tokens during training and the beginning of long inputs during inference.

## Setup in VS Code (Windows)

Open this folder in VS Code. In the integrated PowerShell terminal, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you cloned this repository, run `git lfs pull` to fetch the trained model stored with Git LFS.

## Prepare data and train

The raw public dataset files are included. To regenerate the balanced splits and train:

```powershell
python download_data.py
python prepare_data.py
python train.py --epochs 2
```

The default preparation creates **800 training, 100 validation, and 100 test examples per domain** (2,400 / 300 / 300 total). Legal uses LEDGAR's official train/validation/test partitions. Finance uses Financial PhraseBank's train/test partitions and a held-out slice from its training partition for validation. Healthcare uses a seeded 80/10/10 split of the expert-labeled PubMedQA examples. The exact counts and sources are written to `data/processed/metadata.json` and the training report.

Optional training controls:

```powershell
python train.py --epochs 3 --max-train-per-class 800 --max-eval-per-class 100 --batch-size 8
```

Training saves the model, tokenizer, `metrics.json`, `label_map.json`, and `confusion_matrix.svg` to `artifacts/domain-distilbert/`. The first training run downloads the pretrained checkpoint. A CUDA GPU is recommended; CPU training also works but takes longer.

## Run the app

```powershell
python run_app.py --global.developmentMode false --server.port 8501 --server.address 127.0.0.1
```

Open the URL printed in the terminal. Keep that terminal running while you use the app. If port 8501 is busy, change the port in the command and open the new URL.

## Evaluation

The training script reports accuracy, macro F1, weighted F1, per-domain F1/support, and a confusion matrix on the held-out test split. Because the domains come from different source datasets, scores measure this particular mix and split; they are not a universal benchmark for document-domain classification. Review the per-domain results and inspect misclassified examples before reporting the result.

## Dataset attribution and limitations

See [DATASETS.md](DATASETS.md) for source links, licenses, and citations. The classifier always chooses one of its three known domains, even for unrelated text, and can be influenced by the datasets' different writing styles. Do not use its predictions for high-stakes decisions.
