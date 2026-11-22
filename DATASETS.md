# Dataset sources and attribution

This project maps every text example to a broad source domain. The source datasets' original task labels are not used as model targets.

| Domain | Dataset | Examples used | License / attribution |
|---|---|---:|---|
| Legal | [LexGLUE / LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) | 800 train + 100 validation + 100 test | CC BY 4.0. Cite Chalkidis et al. (2022) and Tuggener et al. (2020). |
| Finance | [Financial PhraseBank](https://huggingface.co/datasets/mteb/financial_phrasebank) | 800 train + 100 validation + 100 test | CC BY-NC-SA 3.0. Cite Malo et al. (2014). Noncommercial and share-alike terms apply. |
| Healthcare | [PubMedQA, pqa_labeled](https://huggingface.co/datasets/highnote/pubmed_qa) | 800 train + 100 validation + 100 test | MIT. Cite Jin et al. (2019). |

## Processing

- LEDGAR uses its official train, validation, and test partitions.
- Financial PhraseBank uses its published train and test partitions. A non-overlapping 100-example slice of the train partition is held out for validation.
- PubMedQA's 1,000 expert-labeled examples have no official train/validation/test partition in this export. They are shuffled with seed 42 and divided 80/10/10.
- The target label is the source domain (`legal`, `finance`, or `healthcare`). Financial sentiment and PubMedQA answer labels are ignored.
- Preprocessing normalizes whitespace and removes control characters. For PubMedQA, the question, abstract contexts, and long answer are concatenated.

## Citations

```text
Chalkidis, I. et al. (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. ACL 2022.
Tuggener, D. et al. (2020). LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts. LREC 2020.
Malo, P. et al. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65.
Jin, Q. et al. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP-IJCNLP 2019.
```
