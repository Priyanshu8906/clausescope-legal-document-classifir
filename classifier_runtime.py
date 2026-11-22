"""Minimal TorchScript/WordPiece runtime for Windows App Control environments.

This keeps document inference independent of Transformers' compiled ``regex``
extension, which some managed Windows policies block.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

LOCAL_PACKAGES = Path(__file__).resolve().parent / ".runtime"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

BATCH_SIZE = 8


def _is_chinese_char(codepoint: int) -> bool:
    return (
        0x4E00 <= codepoint <= 0x9FFF or 0x3400 <= codepoint <= 0x4DBF
        or 0x20000 <= codepoint <= 0x2A6DF or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F or 0x2B820 <= codepoint <= 0x2CEAF
        or 0xF900 <= codepoint <= 0xFAFF or 0x2F800 <= codepoint <= 0x2FA1F
    )


def _basic_tokens(text: str) -> list[str]:
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if char in "\t\n\r" or category == "Zs":
            cleaned.append(" ")
        elif category.startswith("C"):
            continue
        elif _is_chinese_char(ord(char)):
            cleaned.extend((" ", char, " "))
        else:
            cleaned.append(char)
    normalized = unicodedata.normalize("NFD", "".join(cleaned).lower())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    result = []
    for word in normalized.split():
        current = []
        for char in word:
            if unicodedata.category(char).startswith("P") or char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                if current:
                    result.append("".join(current))
                    current = []
                result.append(char)
            else:
                current.append(char)
        if current:
            result.append("".join(current))
    return result


def _wordpiece(token: str, vocab: dict[str, int], unknown_id: int) -> list[int]:
    if len(token) > 100:
        return [unknown_id]
    pieces = []
    start = 0
    while start < len(token):
        end = len(token)
        piece_id = None
        while start < end:
            piece = token[start:end] if start == 0 else "##" + token[start:end]
            if piece in vocab:
                piece_id = vocab[piece]
                break
            end -= 1
        if piece_id is None:
            return [unknown_id]
        pieces.append(piece_id)
        start = end
    return pieces


def encode_windows(text: str, vocab: dict[str, int], specials: dict[str, int],
                   max_length: int, overlap: int) -> tuple[list[list[int]], list[list[int]]]:
    chunk_tokens = max_length - 2
    chunk_step = max(1, chunk_tokens - overlap)
    token_ids = []
    for token in _basic_tokens(text):
        token_ids.extend(_wordpiece(token, vocab, specials["unk"]))
    windows, masks = [], []
    for start in range(0, max(len(token_ids), 1), chunk_step):
        content = token_ids[start:start + chunk_tokens]
        ids = [specials["cls"], *content, specials["sep"]]
        mask = [1] * len(ids)
        padding = max_length - len(ids)
        ids.extend([specials["pad"]] * padding)
        mask.extend([0] * padding)
        windows.append(ids)
        masks.append(mask)
        if start + chunk_tokens >= len(token_ids):
            break
    return windows, masks


class DomainClassifier:
    def __init__(self, model_dir: str | Path):
        import torch

        self.torch = torch
        model_dir = Path(model_dir)
        self.model = torch.jit.load(str(model_dir / "classifier.pt"), map_location="cpu")
        self.model.eval()
        tokenizer = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
        self.vocab = tokenizer["model"]["vocab"]
        self.specials = {
            "pad": self.vocab.get("[PAD]", 0),
            "unk": self.vocab.get("[UNK]", 100),
            "cls": self.vocab.get("[CLS]", 101),
            "sep": self.vocab.get("[SEP]", 102),
        }
        self.labels = json.loads((model_dir.parent / "label_map.json").read_text(encoding="utf-8"))["labels"]
        inference_config = json.loads((model_dir / "inference_config.json").read_text(encoding="utf-8"))
        self.max_length = int(inference_config["max_length"])
        self.chunk_overlap = int(inference_config.get("chunk_overlap", 64))

    def predict(self, text: str) -> tuple[list[tuple[str, float]], int]:
        input_ids, attention_masks = encode_windows(
            text, self.vocab, self.specials, self.max_length, self.chunk_overlap
        )
        probabilities = []
        with self.torch.inference_mode():
            for start in range(0, len(input_ids), BATCH_SIZE):
                ids = self.torch.tensor(input_ids[start:start + BATCH_SIZE], dtype=self.torch.long)
                masks = self.torch.tensor(attention_masks[start:start + BATCH_SIZE], dtype=self.torch.long)
                logits = self.model(ids, masks)
                probabilities.append(self.torch.softmax(logits, dim=-1))
        scores = self.torch.cat(probabilities, dim=0).mean(dim=0).tolist()
        results = sorted(
            [(self.labels[index], float(score)) for index, score in enumerate(scores)],
            key=lambda item: item[1],
            reverse=True,
        )
        return results, len(input_ids)
