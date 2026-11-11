from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class AIDetector:
    """Loads an operator-trained Hugging Face sequence classifier if present."""
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.available = False
        if self.model_path.exists() and (self.model_path / "config.json").exists():
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                self.torch = torch
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
                self.model.eval()
                self.available = True
            except Exception as exc:
                logger.exception("Could not load AI classifier: %s", exc)

    def predict(self, text: str) -> dict:
        if not self.available:
            return {
                "available": False,
                "ai_probability": None,
                "human_probability": None,
                "label": "unavailable",
                "message": "Model AI belum dipasang. Latih model menggunakan dataset berlabel lalu atur AI_MODEL_PATH.",
            }
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1)[0].cpu().tolist()
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        ai_idx = next((i for i, label in id2label.items() if "ai" in label or "machine" in label), 1 if len(probs) > 1 else 0)
        human_idx = next((i for i, label in id2label.items() if "human" in label or "person" in label), 1 - ai_idx if len(probs) == 2 else None)
        ai_p = float(probs[ai_idx])
        human_p = float(probs[human_idx]) if human_idx is not None and human_idx < len(probs) else None
        return {
            "available": True,
            "ai_probability": round(ai_p * 100, 2),
            "human_probability": round(human_p * 100, 2) if human_p is not None else None,
            "label": "ai_likely" if ai_p >= 0.5 else "human_likely",
            "model": str(self.model_path.name),
            "message": "Estimasi model; bukan kepastian dan perlu validasi pada domain serta bahasa dokumen.",
        }
