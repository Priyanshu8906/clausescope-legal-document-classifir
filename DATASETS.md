# Dataset sources and attribution

The classifier uses 12 topic labels. General English web documents come from the `agentlans/en-document-topic-classification` corpus, which derives topic tags from upstream C4 pages and document-classification annotations. Three focused corpora add examples for legal, healthcare, and finance.

| Model label | General topic label | Focused supplement |
|---|---|---|
| Legal | Law & Government | LexGLUE / LEDGAR contract provisions |
| Healthcare | Health & Wellness | PubMedQA expert-labeled biomedical questions, abstracts, and answers |
| Finance | Finance & Investment | Financial PhraseBank news sentences |
| Education | Education & Careers | — |
| Business | Business & Industry | — |
| Manufacturing | Industrial & Manufacturing | — |
| Technology | Software Engineering | — |
| Science | Science & Research | — |
| News & politics | News & Politics | — |
| Sports | Sports & Athletics | — |
| Automotive | Autos & Vehicles | — |
| Arts & entertainment | Arts & Entertainment | — |

## Sources and terms

- [English Document Topic Classification](https://huggingface.co/datasets/agentlans/en-document-topic-classification): 12 selected topic configurations, with 180 training, 40 validation, and 40 held-out test examples per topic. Distributed under the Open Data Commons Attribution License (ODC-BY). Its topic labels come from an upstream annotation pipeline and may be noisy. Cite the dataset card and its upstream sources, `allenai/c4` and `agentlans/en-document-classification`.
- [LexGLUE / LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue): legal contract provisions, supplemental 80/20/20 train/validation/test examples. CC BY 4.0. Cite Chalkidis et al. (2022) and Tuggener et al. (2020).
- [Financial PhraseBank](https://huggingface.co/datasets/mteb/financial_phrasebank): finance-news sentences, supplemental 80/20/20 train/validation/test examples. The source card states CC BY-NC-SA 3.0; noncommercial and share-alike terms apply. Cite Malo et al. (2014).
- [PubMedQA, pqa_labeled](https://huggingface.co/datasets/highnote/pubmed_qa): expert-labeled biomedical questions with abstracts and answers, supplemental 80/20/20 train/validation/test examples from a seeded 80/10/10 split. MIT. Cite Jin et al. (2019).

## Preparation and evaluation split

- The general topic source uses its training configurations for the train and validation sets. A deterministic 180/40 sample is used per topic after filtering out examples shorter than 180 characters.
- General-topic test examples come from the source's separate, stratified `All/test` split; 40 examples are sampled per topic.
- The focused legal, finance, and healthcare supplements use their published partitions or a seeded split. The focused test items are held apart from the corresponding training items.
- Text whitespace and control characters are normalized. Prepared rows retain a `source` field. PubMedQA question, abstract contexts, and long answer are concatenated.
- The raw topic Parquet files are downloaded locally by `download_data.py` and are not tracked in this repository. The smaller prepared splits are included to make the experiment reproducible without re-downloading raw sources.

## Citations

```text
Chalkidis, I. et al. (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. ACL 2022.
Tuggener, D. et al. (2020). LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts. LREC 2020.
Malo, P. et al. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65.
Jin, Q. et al. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP-IJCNLP 2019.
agentlans. English Document Topic Classification Dataset. https://huggingface.co/datasets/agentlans/en-document-topic-classification
``` 
