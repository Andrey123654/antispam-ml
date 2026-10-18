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
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from category import predict as predict_category  # noqa: E402
from train import predict_bundle  # noqa: E402

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "artifacts"))
THRESHOLD = float(os.environ.get("SPAM_THRESHOLD", "0.5"))

app = FastAPI(
    title="MKB251-RazhnovAP · антиспам-фильтр",
    description="Классификация писем как спам/не спам и определение категории письма",
    version="1.0.0",
)

state: dict = {}


@app.on_event("startup")
def load_models() -> None:
    spam_path = MODEL_DIR / "spam_bundle.joblib"
    cat_path = MODEL_DIR / "category_bundle.joblib"
    if not spam_path.exists():
        raise RuntimeError(f"не найден артефакт модели: {spam_path}")
    state["spam"] = joblib.load(spam_path)
    state["category"] = joblib.load(cat_path) if cat_path.exists() else None
    import json
    metrics_path = MODEL_DIR / "metrics.json"
    state["metrics"] = (json.loads(metrics_path.read_text(encoding="utf-8"))
                        if metrics_path.exists() else {})
    state["threshold"] = THRESHOLD


class Email(BaseModel):
    text: str = Field(..., description="Текст письма (тема и тело)")
    subject: Optional[str] = Field("", description="Тема письма")
    sender: Optional[str] = Field("", description="Адрес отправителя")


class Prediction(BaseModel):
    is_spam: bool
    spam_probability: float
    category: str
    threshold: float
    flags: dict


def _to_frame(emails: List[Email]) -> pd.DataFrame:
    return pd.DataFrame([{
        "text": e.text,
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
    proba = predict_bundle(state["spam"], df)
    cats = (predict_category(state["category"], df["text"].tolist())
            if state.get("category") else ["—"] * len(df))
    thr = state["threshold"]
    out = []
    for p, c, e in zip(proba, cats, emails):
        flags = {
            "urls": int(df.loc[df["text"] == e.text, "text"].str.count(r"https?://").iloc[0]),
            "html": bool(df.loc[df["text"] == e.text, "is_html"].iloc[0]),
        }
        out.append(Prediction(is_spam=bool(p >= thr), spam_probability=round(float(p), 5),
                              category=c, threshold=thr, flags=flags))
    return out


@app.get("/")
def root() -> dict:
    return {
        "service": "антиспам-фильтр",
        "owner": "MKB251-RazhnovAP",
        "version": app.version,
        "endpoints": ["/predict", "/predict_batch", "/health", "/docs"],
        "model": "TF-IDF (слова + символьные n-граммы) + LR/SVM/NB/HistGB + стекинг",
    }


@app.get("/health")
def health() -> dict:
    cv = (state.get("metrics") or {}).get("cv", {}).get("stack", {})
    return {
        "status": "ok" if "spam" in state else "no_model",
        "threshold": state.get("threshold"),
        "category_model": state.get("category") is not None,
        "oof_f1": cv.get("f1"),
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
