"""Download selected topic datasets used by the 12-domain classifier."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
FILES = {
    "data/raw/ledgar/train.parquet": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/main/ledgar/train-00000-of-00001.parquet",
    "data/raw/ledgar/validation.parquet": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/main/ledgar/validation-00000-of-00001.parquet",
    "data/raw/ledgar/test.parquet": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/main/ledgar/test-00000-of-00001.parquet",
    "data/raw/finance_phrasebank/train.parquet": "https://huggingface.co/datasets/mteb/financial_phrasebank/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "data/raw/finance_phrasebank/test.parquet": "https://huggingface.co/datasets/mteb/financial_phrasebank/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    "data/raw/healthcare_pubmedqa/train.parquet": "https://huggingface.co/datasets/highnote/pubmed_qa/resolve/refs%2Fconvert%2Fparquet/pqa_labeled/train/0000.parquet",
}

# One focused train split per topic avoids downloading the full 759 MB corpus.
TOPIC_CONFIGS = (
    "LawGovernment", "HealthWellness", "FinanceInvestment", "EducationCareers",
    "BusinessIndustry", "IndustrialManufacturing", "SoftwareEngineering",
    "ScienceResearch", "NewsPolitics", "SportsAthletics", "AutosVehicles",
    "ArtsEntertainment",
)
TOPIC_BASE = (
    "https://huggingface.co/datasets/agentlans/en-document-topic-classification/"
    "resolve/refs%2Fconvert%2Fparquet"
)
for config in TOPIC_CONFIGS:
    FILES[f"data/raw/document_topics/{config}_train.parquet"] = (
        f"{TOPIC_BASE}/{config}/train/0000.parquet"
    )
FILES["data/raw/document_topics/All_test.parquet"] = f"{TOPIC_BASE}/All/test/0000.parquet"


def main() -> None:
    for relative, url in FILES.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.unlink(missing_ok=True)
            print(f"Downloading {relative}…", flush=True)
            urlretrieve(url, temporary)
            temporary.replace(path)
        print(f"{relative}: {path.stat().st_size / 1_000_000:.1f} MB", flush=True)
    print("Source datasets are ready. Run `python prepare_data.py` next.")


if __name__ == "__main__":
    main()
