# -*- coding: utf-8 -*-
"""
app.py — микросервис антиспам-фильтра.

Эндпоинты:
    GET  /            — краткая информация о сервисе
    GET  /health      — состояние и версия моделей
    POST /predict     — классификация одного письма (спам/не спам + категория)
    POST /predict_batch — пакетная классификация

Запуск локально:
    uvicorn service.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from category import predict as predict_category  # noqa: E402
from train import predict_bundle  # noqa: E402
from adapt_hackathon import predict_bundle as predict_hackathon_bundle  # noqa: E402
from hackathon_adapter import email_url_features  # noqa: E402

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "artifacts"))

app = FastAPI(
    title="MKB251-RazhnovAP · антиспам-фильтр",
    description="Антиспам на основе подхода хакатона: NLP письма и автономные признаки ссылок",
    version="2.0.0",
)

state: dict = {}


@app.on_event("startup")
def load_models() -> None:
    spam_path = MODEL_DIR / "hackathon_v2" / "spam_bundle.joblib"
    cat_path = MODEL_DIR / "category_bundle.joblib"
    if not spam_path.exists():
        raise RuntimeError(f"не найден артефакт модели: {spam_path}")
    state["spam"] = joblib.load(spam_path)
    state["category"] = joblib.load(cat_path) if cat_path.exists() else None
    import json
    metrics_path = MODEL_DIR / "hackathon_v2" / "metrics.json"
    state["metrics"] = (json.loads(metrics_path.read_text(encoding="utf-8"))
                        if metrics_path.exists() else {})
    state["threshold"] = float(os.environ.get("SPAM_THRESHOLD", state["spam"]["threshold"]))
    if not 0 <= state["threshold"] <= 1:
        raise RuntimeError("порог должен лежать в диапазоне от 0 до 1")


class Email(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000, description="Тело письма")
    subject: Optional[str] = Field("", max_length=2000, description="Тема письма")
    sender: Optional[str] = Field("", max_length=1000, description="Адрес отправителя")


class Prediction(BaseModel):
    is_spam: bool
    spam_probability: float
    category: str
    threshold: float
    flags: dict


def _to_frame(emails: List[Email]) -> pd.DataFrame:
    return pd.DataFrame([{
        "text": (e.subject or "") + "\n" + e.text,
        "subject": e.subject or "",
        "sender": e.sender or "",
        "body": e.text,
        "is_html": int("<html" in (e.text or "").lower()),
        "has_attach": 0,
        "n_parts": 1,
    } for e in emails])


def _predict(emails: List[Email]) -> List[Prediction]:
    if "spam" not in state:
        raise HTTPException(status_code=503, detail="модель не загружена")
    df = _to_frame(emails)
    proba = predict_hackathon_bundle(state["spam"], df)
    cats = (predict_category(state["category"], df["text"].tolist())
            if state.get("category") else ["—"] * len(df))
    thr = state["threshold"]
    out = []
    for i, (p, c, e) in enumerate(zip(proba, cats, emails)):
        url_features = email_url_features(df.iloc[i]["text"])
        flags = {
            "urls": int(url_features["hack_url_count"]),
            "html": bool(df.iloc[i]["is_html"]),
            "brand_mimicry": bool(url_features["hack_max_brand_mimicry"]),
            "userinfo_in_url": bool(url_features["hack_max_has_userinfo"]),
            "network_requests": False,
        }
        out.append(Prediction(is_spam=bool(p >= thr), spam_probability=round(float(p), 5),
                              category=c, threshold=thr, flags=flags))
    return out


@app.get("/", response_class=HTMLResponse)
def demo():
    return (ROOT / "service" / "demo.html").read_text(encoding="utf-8")


@app.get("/info")
def root() -> dict:
    return {
        "service": "антиспам-фильтр",
        "owner": "MKB251-RazhnovAP",
        "version": app.version,
        "endpoints": ["/predict", "/predict_batch", "/health", "/docs"],
        "model": "TF-IDF + 94 признака письма + 50 признаков ссылок из хакатона + OOF-стекинг",
        "basis": "https://github.com/Andrey123654/Hakaton",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if "spam" in state else "no_model",
        "threshold": state.get("threshold"),
        "category_model": state.get("category") is not None,
        "model_kind": state.get("spam", {}).get("kind"),
        "metrics_threshold": state.get("metrics", {}).get("threshold"),
        "threshold_overridden": "SPAM_THRESHOLD" in os.environ,
    }


@app.post("/predict", response_model=Prediction)
def predict_one(email: Email) -> Prediction:
    return _predict([email])[0]


@app.post("/predict_batch", response_model=List[Prediction])
def predict_many(emails: List[Email]) -> List[Prediction]:
    if not emails:
        raise HTTPException(status_code=400, detail="пустой список писем")
    if len(emails) > 256:
        raise HTTPException(status_code=400, detail="не более 256 писем за запрос")
    return _predict(emails)
