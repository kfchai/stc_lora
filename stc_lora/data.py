"""Corpus utilities: cached TinyShakespeare download."""

from __future__ import annotations

from pathlib import Path

_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
_CACHE_DIR = Path.home() / ".cache" / "stc_lora"


class CorpusLoader:
    """Minimal loader for the TinyShakespeare benchmark domain."""

    @staticmethod
    def load_shakespeare(cache_dir: Path | None = None) -> str:
        """Download TinyShakespeare (~1.1MB) once and return it as a string."""
        cache = cache_dir or _CACHE_DIR
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / "tinyshakespeare.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        import urllib.request
        urllib.request.urlretrieve(_TINY_SHAKESPEARE_URL, path)
        return path.read_text(encoding="utf-8")
