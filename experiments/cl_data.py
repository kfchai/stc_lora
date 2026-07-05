"""Offline 3-domain corpus loader for the continual-learning benchmark.

Three genuinely distinct registers, all available offline (no network):
  - news   : AG News          (modern journalistic prose)
  - wiki   : WikiText-2       (encyclopedic reference prose)
  - shake  : TinyShakespeare  (16th-century dramatic verse; cached by the repo)

Each domain is tokenized against the model's tokenizer and cut into fixed-
length chunks, then split into train / test streams. The strong register
shift (modern news -> encyclopedia -> Elizabethan verse) is what makes
catastrophic forgetting measurable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from stc_lora.data import CorpusLoader

DOMAIN_ORDER = ["news", "wiki", "shake"]


@dataclass
class DomainData:
    name: str
    train_chunks: list[list[int]]   # token-id chunks
    test_chunks: list[list[int]]


def _raw_texts(name: str, max_docs: int) -> list[str]:
    """Return a list of raw text documents for a domain (offline)."""
    if name == "news":
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/ag_news", split=f"train[:{max_docs}]")
        return [t for t in ds["text"] if len(t.strip()) > 40]
    if name == "wiki":
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split=f"train[:{max_docs * 4}]")
        # WikiText has many blank/heading lines; keep substantive paragraphs.
        return [t.strip() for t in ds["text"] if len(t.strip()) > 120][:max_docs]
    if name == "shake":
        text = CorpusLoader.load_shakespeare()
        # Split into paragraph-ish blocks on blank lines.
        blocks = [b.strip().replace("\n", " ") for b in text.split("\n\n")]
        return [b for b in blocks if len(b) > 120][:max_docs]
    raise ValueError(f"unknown domain {name}")


def _chunk_tokens(texts: list[str], tokenizer, chunk_len: int,
                  max_chunks: int) -> list[list[int]]:
    """Concatenate texts, tokenize, and cut into fixed-length chunks."""
    ids: list[int] = []
    for t in texts:
        ids.extend(tokenizer(t).input_ids)
        if len(ids) >= (max_chunks + 1) * chunk_len:
            break
    chunks = [
        ids[i:i + chunk_len]
        for i in range(0, len(ids) - chunk_len + 1, chunk_len)
    ]
    return chunks[:max_chunks]


def load_domains(
    tokenizer,
    chunk_len: int = 128,
    train_chunks: int = 240,
    test_chunks: int = 60,
    max_docs: int = 4000,
) -> list[DomainData]:
    """Load all three domains as tokenized chunk streams.

    Args:
        tokenizer: HF tokenizer (from the model under test).
        chunk_len: tokens per chunk.
        train_chunks / test_chunks: chunks per domain for train / held-out test.
        max_docs: cap on raw documents pulled per domain before chunking.

    Returns:
        List of DomainData in DOMAIN_ORDER.
    """
    out = []
    need = train_chunks + test_chunks
    for name in DOMAIN_ORDER:
        texts = _raw_texts(name, max_docs)
        chunks = _chunk_tokens(texts, tokenizer, chunk_len, need)
        if len(chunks) < need:
            raise RuntimeError(
                f"domain {name}: only {len(chunks)} chunks, need {need}. "
                f"Increase max_docs."
            )
        out.append(DomainData(
            name=name,
            train_chunks=chunks[:train_chunks],
            test_chunks=chunks[train_chunks:train_chunks + test_chunks],
        ))
    return out
