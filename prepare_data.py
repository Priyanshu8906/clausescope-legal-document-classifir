"""Create balanced domain-labeled JSONL splits from the three source datasets."""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = ROOT / ".runtime"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import pyarrow.parquet as pq

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 42
TRAIN_PER_DOMAIN = 800
EVAL_PER_DOMAIN = 100


def rows(path: Path, columns: list[str]) -> list[dict]:
    parquet = pq.ParquetFile(path)
    return [row for group in range(parquet.num_row_groups)
            for row in parquet.read_row_group(group, columns=columns).to_pylist()]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text or "")).strip()


def sample(source: list[dict], count: int, rng: random.Random) -> list[dict]:
    if count > len(source):
        raise ValueError(f"Asked for {count} examples, but only {len(source)} are available")
    return rng.sample(source, count)


def write_split(name: str, examples: list[dict]) -> None:
    path = OUT / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in examples), encoding="utf-8")
    print(f"{name}: {len(examples):,} examples → {path}")


def main() -> None:
    rng = random.Random(SEED)
    counts = {"train": Counter(), "validation": Counter(), "test": Counter()}
    splits: dict[str, list[dict]] = {key: [] for key in counts}

    # Legal: LEDGAR's official train/validation/test partitions.
    legal_sources = {part: rows(RAW / "ledgar" / f"{part}.parquet", ["text"])
                     for part in ("train", "validation", "test")}
    for part, requested in (("train", TRAIN_PER_DOMAIN), ("validation", EVAL_PER_DOMAIN), ("test", EVAL_PER_DOMAIN)):
        chosen = sample(legal_sources[part], requested, rng)
        for row in chosen:
            splits[part].append({"text": clean(row["text"]), "label": "legal"})
            counts[part]["legal"] += 1

    # Finance: Financial PhraseBank's official train and test splits.
    finance_train = rows(RAW / "finance_phrasebank" / "train.parquet", ["text"])
    finance_test = rows(RAW / "finance_phrasebank" / "test.parquet", ["text"])
    shuffled_finance = sample(finance_train, len(finance_train), rng)
    finance_train_samples = shuffled_finance[:TRAIN_PER_DOMAIN]
    finance_validation = shuffled_finance[TRAIN_PER_DOMAIN:TRAIN_PER_DOMAIN + EVAL_PER_DOMAIN]
    for part, source, requested in (
        ("train", finance_train_samples, TRAIN_PER_DOMAIN),
        ("validation", finance_validation, EVAL_PER_DOMAIN),
        ("test", finance_test, EVAL_PER_DOMAIN),
    ):
        chosen = sample(source, requested, rng) if part != "validation" else source
        for row in chosen:
            splits[part].append({"text": clean(row["text"]), "label": "finance"})
            counts[part]["finance"] += 1

    # Healthcare: expert-labeled PubMedQA contexts; create a seeded 80/10/10 split.
    medical_rows = rows(RAW / "healthcare_pubmedqa" / "train.parquet", ["question", "context", "long_answer"])
    rng.shuffle(medical_rows)
    medical_splits = {
        "train": medical_rows[:TRAIN_PER_DOMAIN],
        "validation": medical_rows[TRAIN_PER_DOMAIN:TRAIN_PER_DOMAIN + EVAL_PER_DOMAIN],
        "test": medical_rows[TRAIN_PER_DOMAIN + EVAL_PER_DOMAIN:TRAIN_PER_DOMAIN + 2 * EVAL_PER_DOMAIN],
    }
    for part, source in medical_splits.items():
        for row in source:
            context = row.get("context") or {}
            abstract = " ".join(context.get("contexts", [])) if isinstance(context, dict) else str(context)
            text = " ".join((row.get("question", ""), abstract, row.get("long_answer", "")))
            splits[part].append({"text": clean(text), "label": "healthcare"})
            counts[part]["healthcare"] += 1

    # Shuffle mixed domains so batches are not ordered by label.
    for part in splits:
        rng.shuffle(splits[part])
        write_split(part, splits[part])
    metadata = {
        "seed": SEED,
        "labels": ["legal", "healthcare", "finance"],
        "source_counts": {part: dict(value) for part, value in counts.items()},
        "sources": {
            "legal": "coastalcph/lex_glue, ledgar; official train/validation/test partitions",
            "healthcare": "highnote/pubmed_qa, pqa_labeled; seeded 80/10/10 split",
            "finance": "mteb/financial_phrasebank, default; official train/test partitions, training partition split for validation",
        },
        "preprocessing": "Remove control characters, normalize whitespace, concatenate PubMedQA question, abstract contexts, and long answer.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Per-domain counts:", json.dumps(metadata["source_counts"], indent=2))


if __name__ == "__main__":
    main()
