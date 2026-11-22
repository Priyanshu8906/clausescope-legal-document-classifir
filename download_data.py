"""Download the public legal, finance, and healthcare text datasets."""
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


def main() -> None:
    for relative, url in FILES.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"Downloading {relative}…", flush=True)
            urlretrieve(url, path)
        print(f"{relative}: {path.stat().st_size / 1_000_000:.2f} MB", flush=True)
    print("All source datasets are ready. Run `python prepare_data.py` next.")


if __name__ == "__main__":
    main()
