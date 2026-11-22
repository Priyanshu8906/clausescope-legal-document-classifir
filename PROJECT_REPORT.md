# DomainDoc NLP project report

## Objective

Classify document text into the broad domains **legal**, **healthcare**, and **finance**. The app accepts pasted text or `.txt`, `.pdf`, and `.docx` files, shows the predicted domain and model confidence, and gives a short extractive summary.

## Data

The training set combines three public sources, with the source domain used as the target label:

- LEDGAR legal contract provisions for `legal`.
- Financial PhraseBank financial-news sentences for `finance`.
- Expert-labeled PubMedQA questions, abstract contexts, and answers for `healthcare`.

The prepared split is balanced: 800 examples per domain for training, 100 per domain for validation, and 100 per domain for testing. Legal examples use LEDGAR's official partitions. Finance examples use Financial PhraseBank's official train/test partitions, with a disjoint validation subset from its training partition. PubMedQA's expert-labeled 1,000-example split is randomly divided 80/10/10 with seed 42. See `DATASETS.md` for source links and license notes; `data/processed/metadata.json` records exact counts.

## Preprocessing and model

Preprocessing removes control characters and normalizes whitespace. For PubMedQA, the question, abstract context, and long answer are joined into one text field. The classifier fine-tunes `distilbert/distilbert-base-uncased` with a three-class classification head. The run uses a maximum sequence length of 192 tokens, batch size 8, two epochs, and seed 42.

## Evaluation

The model was fine-tuned for two epochs on CPU. On the held-out 300-example test split (100 examples per domain), it achieved:

| Metric | Score |
|---|---:|
| Accuracy | 99.33% |
| Macro F1 | 99.33% |
| Weighted F1 | 99.33% |
| Legal F1 | 99.01% |
| Healthcare F1 | 100.00% |
| Finance F1 | 98.99% |

The model correctly classified all legal and healthcare test items and misclassified two finance items as legal. The exact run metrics are in `artifacts/domain-distilbert/metrics.json`; the confusion matrix is `artifacts/domain-distilbert/confusion_matrix.svg`.

The three source datasets differ in length, genre, and collection process. As a result, the model may learn source-specific style cues as well as domain content. Report these results as a classroom experiment on this dataset mixture, not as a general benchmark for arbitrary documents.

## Limitations and appropriate use

This is an educational prototype, not a legal, medical, or financial advice system. It always predicts one of its three known categories, including for unrelated content. The softmax score is not calibrated confidence. Long documents are truncated to the model's input limit, so only their beginning may influence the prediction.

## Reproduction

```powershell
python download_data.py
python prepare_data.py
python train.py --epochs 2
python run_app.py --global.developmentMode false --server.port 8501 --server.address 127.0.0.1
```

The first fine-tuning run downloads the base checkpoint. Raw datasets and the trained model are stored in the project repository.

## Citations

- Chalkidis et al. (2022), *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*, ACL 2022.
- Tuggener et al. (2020), *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts*, LREC 2020.
- Malo et al. (2014), *Good debt or bad debt: Detecting semantic orientations in economic texts*, JASIST 65.
- Jin et al. (2019), *PubMedQA: A Dataset for Biomedical Research Question Answering*, EMNLP-IJCNLP 2019.
