from __future__ import annotations
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityIndex:
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir
        self.documents: list[dict] = []
        self.vectorizer = None
        self.matrix = None
        self.reload()

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def reload(self) -> None:
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        docs = []
        for path in sorted(self.corpus_dir.rglob("*.txt")):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore").strip()
                if len(content) >= 30:
                    docs.append({"id": path.relative_to(self.corpus_dir).as_posix(), "text": content})
            except OSError:
                continue
        self.documents = docs
        if not docs:
            self.vectorizer, self.matrix = None, None
            return
        corpus = [d["text"] for d in docs]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            analyzer="word",
            sublinear_tf=True,
            max_features=250_000,
        )
        try:
            self.matrix = self.vectorizer.fit_transform(corpus)
        except ValueError:
            self.vectorizer, self.matrix = None, None

    def search(self, text: str, top_k: int = 5) -> list[dict]:
        if self.vectorizer is None or self.matrix is None:
            return []
        query = self.vectorizer.transform([text])
        scores = cosine_similarity(query, self.matrix)[0]
        order = scores.argsort()[::-1][:top_k]
        results = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append({
                "source_id": self.documents[idx]["id"],
                "similarity_percent": round(score * 100, 2),
                "method": "TF-IDF cosine similarity",
            })
        return results
