from __future__ import annotations
import re
import math
from collections import Counter
from typing import List

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+", re.UNICODE)
CONNECTORS = {
    "oleh karena itu", "namun", "tetapi", "akan tetapi", "selain itu", "di samping itu",
    "sebab", "karena", "sehingga", "meskipun", "walaupun", "dengan demikian",
    "pertama", "kedua", "akhirnya", "sementara itu", "di sisi lain", "misalnya",
}


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_RE.split(text.strip()) if s.strip()]


def analyze_linguistics(text: str, sentences: List[str]) -> dict:
    words = [w.lower() for w in WORD_RE.findall(text)]
    lengths = [len(WORD_RE.findall(s)) for s in sentences]
    count = len(words)
    mean = sum(lengths) / len(lengths) if lengths else 0.0
    variance = sum((n - mean) ** 2 for n in lengths) / len(lengths) if lengths else 0.0
    std = math.sqrt(variance)
    unique_ratio = len(set(words)) / count if count else 0.0
    lower = text.lower()
    found_connectors = sorted({c for c in CONNECTORS if re.search(r"(?<!\w)" + re.escape(c) + r"(?!\w)", lower)})
    long_sentences = sum(1 for n in lengths if n >= 30)
    short_sentences = sum(1 for n in lengths if n <= 7)
    return {
        "word_count": count,
        "sentence_count": len(sentences),
        "average_sentence_words": round(mean, 2),
        "sentence_length_stddev": round(std, 2),
        "lexical_diversity": round(unique_ratio, 3),
        "short_sentence_count": short_sentences,
        "long_sentence_count": long_sentences,
        "connectors_found": found_connectors,
        "connector_count": len(found_connectors),
        "interpretation": "Deskripsi statistik linguistik; bukan bukti bahwa teks ditulis manusia atau AI.",
    }
