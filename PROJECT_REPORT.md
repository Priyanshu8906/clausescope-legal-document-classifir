# DomainDesk NLP project report

## Objective

Build a document-topic classifier that predicts a broad domain from text. The app supports pasted text and `.txt`, `.pdf`, and `.docx` files, returns a ranked list of domain matches, and displays a short extractive preview.

## Supported categories

The classifier predicts one of 12 categories: legal, healthcare, finance, education, business, manufacturing, technology, science, news and politics, sports, automotive, or arts and entertainment.

## Data and preparation

The main source is the English Document Topic Classification dataset, which contains web documents with 27 topic labels. This project uses 12 of those categories. It uses 180 train and 40 validation examples per category from the topic-specific training partitions, and 40 held-out examples per category from the separate test partition. Examples shorter than 180 characters are excluded from this source.

Three focused corpora add 80 train, 20 validation, and 20 test examples apiece to legal, healthcare, and finance:

- LEDGAR contract provisions for legal.
- PubMedQA questions, abstract context, and answers for healthcare.
- Financial PhraseBank news sentences for finance.

The combined prepared set has 2,400 training examples, 540 validation examples, and 540 test examples. Training uses class-balanced sampling so the supplemental rows do not overwhelm the other categories. Detailed source counts are in `data/processed/metadata.json`; source and license notes are in `DATASETS.md`.

## Model and interface

The classifier fine-tunes `distilbert/distilbert-base-uncased` for two epochs with seed 42 and a maximum training sequence length of 192 tokens. At inference, the app divides long text into overlapping 192-token windows and averages their class scores. The Streamlit interface includes sample documents, file upload or text entry, the leading domain matches, and an extractive preview.

## Held-out evaluation

The two-epoch CPU run scored **89.63% accuracy**, **89.49% macro F1**, and **89.46% weighted F1** on the 540-example test mixture. Per-category F1 is:

| Category | Test examples | F1 |
|---|---:|---:|
| Legal | 60 | 88.89% |
| Healthcare | 60 | 94.92% |
| Finance | 60 | 84.03% |
| Education & careers | 40 | 90.24% |
| Business & industry | 40 | 74.29% |
| Industrial & manufacturing | 40 | 87.67% |
| Software & technology | 40 | 94.87% |
| Science & research | 40 | 93.98% |
| News & politics | 40 | 98.77% |
| Sports & athletics | 40 | 83.78% |
| Autos & vehicles | 40 | 88.89% |
| Arts & entertainment | 40 | 93.51% |

Business and finance were the most confused pair in this run; sports and automotive examples also overlapped. The complete metrics and confusion matrix are saved in `artifacts/domain-distilbert/metrics.json` and `artifacts/domain-distilbert/confusion_matrix.svg`.

## Limitations

The general topic labels are inherited from an upstream annotation pipeline and can be noisy. The sources include English web pages, legal provisions, biomedical QA text, and short financial-news sentences, so their style and content differ. These metrics describe this particular held-out mixture; they do not guarantee performance on every user's documents.

The app always selects from its 12 supported categories, even when text is outside that set. Its softmax score is not a calibrated probability. Scanned PDFs require OCR. The first 180,000 characters can be classified, split into overlapping windows.

## Reproduction

```powershell
python download_data.py
python prepare_data.py
python train.py --epochs 2 --max-length 192 --batch-size 8
python run_app.py --global.developmentMode false --server.port 8501 --server.address 127.0.0.1
```

The selected raw topic files total about 530 MB and are downloaded locally; they are not stored in Git. The prepared splits and trained model are stored in the project.

## References

- Chalkidis et al. (2022), *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*, ACL 2022.
- Tuggener et al. (2020), *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts*, LREC 2020.
- Malo et al. (2014), *Good debt or bad debt: Detecting semantic orientations in economic texts*, JASIST 65.
- Jin et al. (2019), *PubMedQA: A Dataset for Biomedical Research Question Answering*, EMNLP-IJCNLP 2019.
- [English Document Topic Classification dataset](https://huggingface.co/datasets/agentlans/en-document-topic-classification) and its upstream source attribution.
