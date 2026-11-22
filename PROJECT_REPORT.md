# ClauseScope project report

## Objective

Build a domain-specific document classifier for legal contract provisions. The model assigns one of eight common clause types and provides a local interface for text or document upload.

## Dataset

The project uses LEDGAR from the LexGLUE benchmark. It contains 80,000 labeled provisions: 60,000 training, 10,000 validation, and 10,000 test rows. The original files are included in `data/raw/ledgar/` as Parquet. Each row has a provision text and one of 100 clause labels. The classifier selects the eight most frequent labels using only the training split.

The downloaded Parquet splits are full and untouched. To make fine-tuning practical on a student workstation, this run sampled up to 400 training examples per selected label and up to 250 validation/test examples per label. Sampling used seed 42. Exact counts are recorded in `artifacts/ledgar-distilbert/metrics.json`.

## Preprocessing and model

Text preprocessing removes control characters and normalizes whitespace while retaining legal punctuation and section references. The project fine-tunes `distilbert/distilbert-base-uncased` with a single-label sequence-classification head. It trained for two epochs on the CPU with a maximum sequence length of 192 and a batch size of 8.

## Results

On the sampled, held-out test subset (1,891 provisions):

- Accuracy: **98.73%**
- Macro F1: **98.73%**
- Weighted F1: **98.73%**

The confusion matrix is in `artifacts/ledgar-distilbert/confusion_matrix.svg`; per-class F1 and support counts are in `metrics.json`. These results describe this seeded, capped run. They should not be presented as the official full LEDGAR benchmark result.

## Limitations

The dataset contains short U.S. contract provisions, not arbitrary complete documents. This is a closed-set model: it always chooses among its eight known labels, including when the input does not belong to one of them. The GUI warns for long input and truncates it before inference. Its softmax score is confidence, not a calibrated probability. It is an educational prototype and does not provide legal advice.

## Reproduction

```powershell
python download_data.py
python train.py --top-k 8 --epochs 2 --max-train-per-class 400 --max-eval-per-class 250
python run_app.py
```

The first training run downloads the public base checkpoint. The dataset itself is already stored locally.

## References

- Chalkidis et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*. ACL 2022.
- Tuggener et al. (2020). *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts*. LREC 2020.
- Dataset page: `https://huggingface.co/datasets/coastalcph/lex_glue`
- Model page: `https://huggingface.co/distilbert/distilbert-base-uncased`
