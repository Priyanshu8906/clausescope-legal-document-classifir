"""Fine-tune DistilBERT on the downloaded LEDGAR Parquet files."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# On managed Windows machines, some policies block binary wheels from the
# per-user Python directory. This project-local dependency path avoids that.
LOCAL_PACKAGES = ROOT / ".runtime"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

# This managed Windows image blocks NumPy's compiled random extension DLL.
# Torch/Transformers only need NumPy's random submodule for optional graph
# tracing in this project; training randomness is provided by torch itself.
try:
    import numpy as np
    try:
        import numpy.random  # noqa: F401
    except (ImportError, OSError):
        np.random = types.ModuleType("numpy.random")
        np.random.Generator = object
        np.random.BitGenerator = object
        np.random.RandomState = object
        sys.modules["numpy.random"] = np.random
except ImportError:
    pass

import torch
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "distilbert/distilbert-base-uncased"
DATA_DIR = ROOT / "data" / "raw" / "ledgar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=8, help="Most frequent training labels to classify")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-train-per-class", type=int, default=400)
    parser.add_argument("--max-eval-per-class", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "ledgar-distilbert")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python download_data.py` first.")
    parquet = pq.ParquetFile(path)
    return [row for group in range(parquet.num_row_groups)
            for row in parquet.read_row_group(group, columns=["text", "label"]).to_pylist()]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text or "")).strip()


def choose(rows: list[dict], allowed: set[int], label_map: dict[int, int], cap: int, seed: int) -> list[dict]:
    groups: dict[int, list[dict]] = {label: [] for label in allowed}
    for row in rows:
        label = row["label"]
        if label in allowed:
            groups[label].append({"text": clean(row["text"]), "label": label_map[label]})
    chosen = []
    rng = random.Random(seed)
    for original_label in sorted(allowed):
        group = groups[original_label]
        rng.shuffle(group)
        chosen.extend(group[:cap] if cap > 0 else group)
    rng.shuffle(chosen)
    return chosen


def make_loader(samples: list[dict], tokenizer, batch_size: int, max_length: int, shuffle: bool):
    tokenized = tokenizer([x["text"] for x in samples], truncation=True, max_length=max_length, padding=False)
    features = [{"input_ids": tokenized["input_ids"][i], "attention_mask": tokenized["attention_mask"][i],
                 "labels": samples[i]["label"]} for i in range(len(samples))]
    def collate(batch):
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        encoded = tokenizer.pad([{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
                                 for item in batch], padding=True, return_tensors="pt")
        encoded["labels"] = labels
        return encoded
    return DataLoader(features, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def evaluate(model, loader, device, label_count: int) -> tuple[dict, list[list[int]]]:
    confusion = [[0 for _ in range(label_count)] for _ in range(label_count)]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.argmax(dim=-1)
            for expected, predicted in zip(batch["labels"].cpu().tolist(), predictions.cpu().tolist()):
                confusion[expected][predicted] += 1
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(label_count))
    f1s, supports = [], [sum(row) for row in confusion]
    for label in range(label_count):
        tp = confusion[label][label]
        fp = sum(confusion[row][label] for row in range(label_count)) - tp
        fn = supports[label] - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    metrics = {"accuracy": correct / total if total else 0.0,
               "macro_f1": sum(f1s) / label_count,
               "weighted_f1": sum(f * n for f, n in zip(f1s, supports)) / total if total else 0.0,
               "examples": total, "per_label_f1": f1s, "per_label_support": supports}
    return metrics, confusion


def save_confusion_svg(path: Path, matrix: list[list[int]], names: list[str]) -> None:
    n = len(names)
    cell, margin_left, margin_top = 62, 180, 175
    width, height = margin_left + n * cell + 20, margin_top + n * cell + 35
    max_value = max((max(row) for row in matrix), default=1) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="28" font-family="Arial" font-size="18" font-weight="bold">LEDGAR test confusion matrix</text>',
             '<text x="20" y="52" font-family="Arial" font-size="12">Rows: true label · Columns: predicted label · raw counts</text>']
    for i, name in enumerate(names):
        y = margin_top + i * cell + 36
        parts.append(f'<text x="{margin_left-10}" y="{y}" text-anchor="end" font-family="Arial" font-size="11">{name}</text>')
        x = margin_left + i * cell + 30
        parts.append(f'<text x="{x}" y="{margin_top-10}" transform="rotate(-55 {x} {margin_top-10})" font-family="Arial" font-size="11">{name}</text>')
        for j, count in enumerate(matrix[i]):
            intensity = count / max_value
            blue = int(255 - 130 * intensity)
            fill = f"#{int(blue*.72):02x}{int(blue*.85):02x}{blue:02x}"
            cx, cy = margin_left + j * cell, margin_top + i * cell
            parts.append(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
            parts.append(f'<text x="{cx+cell/2}" y="{cy+cell/2+4}" text-anchor="middle" font-family="Arial" font-size="12">{count}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))["label_names"]
    train_rows = load_rows("train")
    counts = Counter(row["label"] for row in train_rows)
    selected = [label for label, _ in counts.most_common(args.top_k)]
    selected_names = [labels[label] for label in selected]
    label_map = {old: new for new, old in enumerate(selected)}
    print("Selected labels (frequency counted on train only):", flush=True)
    for old, name in zip(selected, selected_names):
        print(f"  {name}: {counts[old]:,}", flush=True)
    train_samples = choose(train_rows, set(selected), label_map, args.max_train_per_class, args.seed)
    val_samples = choose(load_rows("validation"), set(selected), label_map, args.max_eval_per_class, args.seed + 1)
    test_samples = choose(load_rows("test"), set(selected), label_map, args.max_eval_per_class, args.seed + 2)
    print(f"Selected samples — train {len(train_samples):,}, validation {len(val_samples):,}, test {len(test_samples):,}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}); loading {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(selected), id2label={i: name for i, name in enumerate(selected_names)},
        label2id={name: i for i, name in enumerate(selected_names)})
    model.to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    train_loader = make_loader(train_samples, tokenizer, args.batch_size, args.max_length, True)
    val_loader = make_loader(val_samples, tokenizer, args.batch_size * 2, args.max_length, False)
    for epoch in range(args.epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = output.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            steps += 1
            if steps % 25 == 0:
                print(f"epoch {epoch+1}/{args.epochs} · step {steps}/{len(train_loader)} · loss {total_loss/steps:.4f}", flush=True)
        val_metrics, _ = evaluate(model, val_loader, device, len(selected))
        print(f"epoch {epoch+1}: train_loss={total_loss/max(steps,1):.4f}, val_accuracy={val_metrics['accuracy']:.4f}, val_macro_f1={val_metrics['macro_f1']:.4f}", flush=True)

    test_loader = make_loader(test_samples, tokenizer, args.batch_size * 2, args.max_length, False)
    test_metrics, confusion = evaluate(model, test_loader, device, len(selected))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (args.output_dir / "label_map.json").write_text(json.dumps({"labels": selected_names}, indent=2), encoding="utf-8")
    report = {"dataset": "coastalcph/lex_glue / ledgar", "base_model": MODEL_ID,
              "device": str(device), "seed": args.seed, "epochs": args.epochs,
              "max_length": args.max_length, "top_k": args.top_k,
              "max_train_per_class": args.max_train_per_class, "max_eval_per_class": args.max_eval_per_class,
              "labels": selected_names,
              "sample_sizes": {"train": len(train_samples), "validation": len(val_samples), "test": len(test_samples)},
              "test": test_metrics, "confusion_matrix": confusion}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion_svg(args.output_dir / "confusion_matrix.svg", confusion, selected_names)
    print("Held-out test metrics:", json.dumps(test_metrics, indent=2), flush=True)
    print(f"Artifacts saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
