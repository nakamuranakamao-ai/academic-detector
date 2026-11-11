from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .nlp import analyze_linguistics, split_sentences
from .similarity import SimilarityIndex
from .ai_detector import AIDetector

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", str(BASE_DIR / "corpus")))
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("academic-detector")

app = FastAPI(
    title="Academic Detector API",
    version="1.0.0",
    description="NLP diagnostics, optional trained AI-text classifier, and corpus similarity search.",
)
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

ai_detector = AIDetector(os.getenv("AI_MODEL_PATH", str(BASE_DIR / "models" / "ai-classifier")))
similarity_index = SimilarityIndex(CORPUS_DIR)


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(BASE_DIR / "frontend.html")


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=30, max_length=200_000)
    include_similarity: bool = True
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ai_model_loaded": ai_detector.available,
        "similarity_documents": similarity_index.document_count,
        "message": "AI classifier requires a fine-tuned model; heuristic indicators are not an AI probability.",
    }


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    text = payload.text.strip()
    if len(text) < 30:
        raise HTTPException(status_code=422, detail="Teks minimal 30 karakter untuk dianalisis.")

    sentences = split_sentences(text)
    linguistic = analyze_linguistics(text, sentences)
    ai_result = ai_detector.predict(text)
    sentence_results = []
    for sentence in sentences:
        sentence_ai = ai_detector.predict(sentence) if ai_detector.available else None
        sentence_results.append({
            "text": sentence,
            "ai_probability": sentence_ai["ai_probability"] if sentence_ai else None,
            "label": sentence_ai["label"] if sentence_ai else "unavailable",
            "word_count": len(re.findall(r"\b[\w'-]+\b", sentence, flags=re.UNICODE)),
        })

    similarity = similarity_index.search(text, payload.top_k) if payload.include_similarity else []
    return {
        "document": {"characters": len(text), "words": linguistic["word_count"], "sentences": len(sentences)},
        "ai_detection": ai_result,
        "linguistic_analysis": linguistic,
        "sentence_analysis": sentence_results,
        "similarity": {
            "available": similarity_index.document_count > 0,
            "corpus_documents": similarity_index.document_count,
            "matches": similarity,
            "method": "TF-IDF word/character n-grams + cosine similarity",
            "scope_notice": "Hanya membandingkan dengan dokumen di corpus lokal yang disediakan operator.",
        },
        "limitations": [
            "Deteksi AI bersifat probabilistik dan dapat salah mengklasifikasikan tulisan manusia maupun AI.",
            "Tanpa model fine-tuned, sistem tidak mengeluarkan skor probabilitas AI.",
            "Similarity bukan pemeriksaan terhadap seluruh internet atau database privat Turnitin.",
        ],
    }


@app.post("/api/corpus/reload")
def reload_corpus() -> dict[str, Any]:
    # Operator places authorized reference .txt files in the corpus directory, then reloads.
    similarity_index.reload()
    return {"status": "ok", "documents_indexed": similarity_index.document_count}


@app.get("/api/corpus/status")
def corpus_status() -> dict[str, Any]:
    return {"documents_indexed": similarity_index.document_count, "corpus_directory": str(CORPUS_DIR)}
