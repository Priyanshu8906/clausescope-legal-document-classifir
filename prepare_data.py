"""Prepare balanced splits from real web documents and domain-specific corpora."""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = ROOT / ".runtime"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import pyarrow.parquet as pq

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 42
GENERAL_TRAIN = 180
GENERAL_VALIDATION = 40
GENERAL_TEST = 40
SUPPLEMENT_TRAIN = 80
SUPPLEMENT_EVAL = 20

# Topic strings are the dataset's published annotation labels.
TOPICS = {
    "LawGovernment": ("legal", "Law & Government"),
    "HealthWellness": ("healthcare", "Health & Wellness"),
    "FinanceInvestment": ("finance", "Finance & Investment"),
    "EducationCareers": ("education", "Education & Careers"),
    "BusinessIndustry": ("business", "Business & Industry"),
    "IndustrialManufacturing": ("manufacturing", "Industrial & Manufacturing"),
    "SoftwareEngineering": ("technology", "Software Engineering"),
    "ScienceResearch": ("science", "Science & Research"),
    "NewsPolitics": ("news_politics", "News & Politics"),
    "SportsAthletics": ("sports", "Sports & Athletics"),
    "AutosVehicles": ("automotive", "Autos & Vehicles"),
    "ArtsEntertainment": ("arts_entertainment", "Arts & Entertainment"),
}
LABELS = [value[0] for value in TOPICS.values()]
DISPLAY = {
    "legal": "Legal",
    "healthcare": "Healthcare",
    "finance": "Finance & investment",
    "education": "Education & careers",
    "business": "Business & industry",
    "manufacturing": "Industrial & manufacturing",
    "technology": "Software & technology",
    "science": "Science & research",
    "news_politics": "News & politics",
    "sports": "Sports & athletics",
    "automotive": "Autos & vehicles",
    "arts_entertainment": "Arts & entertainment",
}


def rows(path: Path, columns: list[str]) -> list[dict]:
    parquet = pq.ParquetFile(path)
    return [
        row
        for group in range(parquet.num_row_groups)
        for row in parquet.read_row_group(group, columns=columns).to_pylist()
    ]


def clean(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def sample(source: list[dict], count: int, rng: random.Random, minimum_chars: int = 0) -> list[dict]:
    eligible = [row for row in source if len(clean(row.get("text", ""))) >= minimum_chars]
    if len(eligible) < count:
        raise ValueError(f"Requested {count} examples with at least {minimum_chars} characters; found {len(eligible)}")
    return rng.sample(eligible, count)


def make_item(text: str, label: str, source: str) -> dict:
    return {"text": clean(text)[:24_000], "label": label, "source": source}


def add(splits: dict[str, list[dict]], counts: dict[str, Counter], part: str,
        examples: list[dict], label: str, source: str) -> None:
    for row in examples:
        item = make_item(row["text"], label, source)
        if item["text"]:
            splits[part].append(item)
            counts[part][label] += 1


def main() -> None:
    rng = random.Random(SEED)
    splits: dict[str, list[dict]] = {part: [] for part in ("train", "validation", "test")}
    counts = {part: Counter() for part in splits}
    sources: dict[str, list[str]] = defaultdict(list)

    # The upstream topic dataset has real English web documents and separate
    # topic configs. A reserved portion of each training config becomes validation;
    # the dataset's disjoint, stratified All/test split provides the held-out test.
    test_rows = rows(RAW / "document_topics" / "All_test.parquet", ["text", "topic"])
    test_by_topic: dict[str, list[dict]] = defaultdict(list)
    for row in test_rows:
        test_by_topic[row["topic"]].append(row)

    for config, (label, topic_name) in TOPICS.items():
        train_rows = rows(RAW / "document_topics" / f"{config}_train.parquet", ["text"])
        train_rows = [row for row in train_rows if len(clean(row.get("text", ""))) >= 180]
        rng.shuffle(train_rows)
        if len(train_rows) < GENERAL_TRAIN + GENERAL_VALIDATION:
            raise ValueError(f"Not enough long examples in {config}: {len(train_rows)}")
        train_selected = train_rows[:GENERAL_TRAIN]
        # Reserve validation rows from the same official training split, disjointly.
        validation_selected = train_rows[GENERAL_TRAIN:GENERAL_TRAIN + GENERAL_VALIDATION]
        test_selected = sample(test_by_topic[topic_name], GENERAL_TEST, rng, minimum_chars=180)
        add(splits, counts, "train", train_selected, label, "agentlans/en-document-topic-classification")
        add(splits, counts, "validation", validation_selected, label, "agentlans/en-document-topic-classification")
        add(splits, counts, "test", test_selected, label, "agentlans/en-document-topic-classification")
        sources[label].append(f"agentlans/en-document-topic-classification:{topic_name}")

    # Add focused expert/domain corpora for the three originally requested areas.
    legal_sources = {
        part: rows(RAW / "ledgar" / f"{part}.parquet", ["text"])
        for part in ("train", "validation", "test")
    }
    for part, count in (("train", SUPPLEMENT_TRAIN), ("validation", SUPPLEMENT_EVAL), ("test", SUPPLEMENT_EVAL)):
        add(splits, counts, part, sample(legal_sources[part], count, rng), "legal", "LexGLUE/LEDGAR")
    sources["legal"].append("coastalcph/lex_glue:ledgar")

    finance_train = rows(RAW / "finance_phrasebank" / "train.parquet", ["text"])
    finance_test = rows(RAW / "finance_phrasebank" / "test.parquet", ["text"])
    rng.shuffle(finance_train)
    add(splits, counts, "train", sample(finance_train[:800], SUPPLEMENT_TRAIN, rng), "finance", "Financial PhraseBank")
    add(splits, counts, "validation", sample(finance_train[800:], SUPPLEMENT_EVAL, rng), "finance", "Financial PhraseBank")
    add(splits, counts, "test", sample(finance_test, SUPPLEMENT_EVAL, rng), "finance", "Financial PhraseBank")
    sources["finance"].append("mteb/financial_phrasebank")

    medical = rows(RAW / "healthcare_pubmedqa" / "train.parquet", ["question", "context", "long_answer"])
    rng.shuffle(medical)
    medical_parts = {"train": medical[:800], "validation": medical[800:900], "test": medical[900:]}
    for part, count in (("train", SUPPLEMENT_TRAIN), ("validation", SUPPLEMENT_EVAL), ("test", SUPPLEMENT_EVAL)):
        examples = []
        for row in sample(medical_parts[part], count, rng):
            context = row.get("context") or {}
            abstract = " ".join(context.get("contexts", [])) if isinstance(context, dict) else str(context)
            examples.append({"text": " ".join((row.get("question", ""), abstract, row.get("long_answer", "")))})
        add(splits, counts, part, examples, "healthcare", "PubMedQA:pqa_labeled")
    sources["healthcare"].append("highnote/pubmed_qa:pqa_labeled")

    for part, examples in splits.items():
        rng.shuffle(examples)
        path = OUT / f"{part}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8")
        print(f"{part}: {len(examples):,} examples → {path}")

    metadata = {
        "seed": SEED,
        "labels": LABELS,
        "display_names": DISPLAY,
        "source_counts": {part: dict(counts[part]) for part in counts},
        "sources": {label: values for label, values in sources.items()},
        "topic_dataset": "agentlans/en-document-topic-classification; 12 focused topic configs; train and held-out All/test partitions",
        "preprocessing": "Normalize whitespace/control characters, remove very short general-topic examples, truncate each row to 24,000 characters. PubMedQA question/context/answer are concatenated.",
        "limitations": "The general topic labels are inherited from an upstream web-document annotation pipeline and may contain label noise. The data are English web documents, which differ from some scanned, multilingual, or highly specialized records. Model scores are not calibrated probabilities.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Per-domain counts:", json.dumps(metadata["source_counts"], indent=2))


if __name__ == "__main__":
    main()
