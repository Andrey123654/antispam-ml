# -*- coding: utf-8 -*-
"""
category.py — дополнительное задание: определение категории письма.

Категории: реклама, предложение работы, знакомства, финансы и инвестиции,
доставка и заказы, безопасность и фишинг, прочее.

Разметки категорий в публичных корпусах нет, поэтому применяется **слабая
разметка правилами по ключевым словам**. Это прозрачно (правила приведены ниже
и в отчёте) и воспроизводимо; ограничение честно фиксируется: модель учится
воспроизводить и обобщать правила, а не экспертные метки.

Запуск:
    python src/category.py --data data/processed/train.csv --model-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import clean_for_tfidf  # noqa: E402

SEED = 42

CATEGORIES = {
    "реклама": r"\bsale\b|\bdiscount\b|\boffer\b|\bpromo\b|\bfree\b|\bdeal\b|скидк|акци|распродаж|предложени[ея]|реклам",
    "работа": r"\bvacanc|\bresume\b|\bsalar|\bjob\b|\bhiring\b|\bemploy|ваканс|резюме|зарплат|работ[ау]|собеседован",
    "знакомства": r"\bdating\b|\bflirt\b|\bsingle\b|\bmeet\b.*\bgirl|\bsexy\b|знакомств|одинок|свидан",
    "финансы": r"\bcrypto|\bbitcoin|\binvest|\bloan\b|\bcredit\b|\bmoney\b|\bprofit|\btrading\b|инвестиц|кредит|прибыл|биткоин|заработ",
    "доставка": r"\bdelivery\b|\btracking\b|\bparcel\b|\bshipping\b|\border\b|доставк|заказ|трек|посылк",
    "безопасность": r"\bverify\b|\bpassword\b|\baccount\b.*\bsuspend|\blogin\b|\bsecurity\b|подтверд|парол|аккаунт|безопасн",
}
COMPILED = {k: re.compile(v, re.I) for k, v in CATEGORIES.items()}


def label_categories(texts: pd.Series) -> np.ndarray:
    """Слабая разметка: категория с наибольшим числом совпадений, иначе «прочее»."""
    labels = []
    for t in texts:
        low = str(t).lower()
        scores = {cat: len(rx.findall(low)) for cat, rx in COMPILED.items()}
        best = max(scores, key=scores.get)
        labels.append(best if scores[best] > 0 else "прочее")
    return np.array(labels)


def build(train: pd.DataFrame, model_dir: Path) -> dict:
    t0 = time.time()
    train = train.copy()
    train["text"] = (train["subject"].fillna("") + "\n" + train["body"].fillna("")).astype(str)
    train["category"] = label_categories(train["text"])
    dist = train["category"].value_counts()
    print("распределение слабых меток:")
    for cat, n in dist.items():
        print(f"  {cat:<14} {n:>5} ({n / len(train) * 100:4.1f} %)")

    # модель учится на всех письмах: и спам, и нормальные (категория у обоих)
    X_text = train["text"].map(clean_for_tfidf)
    y = train["category"].values
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                          strip_accents="unicode", token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(X_text)

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.empty(len(y), dtype=object)
    for itr, iva in skf.split(X, y):
        m = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
        m.fit(X[itr], y[itr])
        oof[iva] = m.predict(X[iva])
    macro = f1_score(y, oof, average="macro")
    report = classification_report(y, oof, output_dict=True, zero_division=0)
    print(f"\nкросс-валидация категорий: macro-F1 = {macro:.4f}")
    for cat in sorted(set(y)):
        r = report.get(cat, {})
        print(f"  {cat:<14} F1={r.get('f1-score', 0):.4f} "
              f"P={r.get('precision', 0):.4f} R={r.get('recall', 0):.4f} n={int(r.get('support', 0))}")

    model = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(X, y)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "model": model}, model_dir / "category_bundle.joblib",
                compress=3)
    out = {
        "macro_f1": round(float(macro), 5),
        "per_class": {c: {k: round(float(v), 4) for k, v in report[c].items()}
                      for c in report if isinstance(report[c], dict) and "f1-score" in report[c]},
        "labeling": "слабая разметка правилами по ключевым словам",
        "distribution": {k: int(v) for k, v in dist.items()},
        "n": int(len(train)),
        "runtime_sec": round(time.time() - t0, 1),
        "confusion": confusion_matrix(y, oof, labels=sorted(set(y))).tolist(),
        "classes": sorted(set(y)),
    }
    (model_dir / "category_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {model_dir}/category_bundle.joblib | macro-F1={macro:.4f}")
    return out


def predict(bundle: dict, texts: list[str]) -> list[str]:
    X = bundle["vectorizer"].transform([clean_for_tfidf(t) for t in texts])
    return list(bundle["model"].predict(X))


def main() -> None:
    ap = argparse.ArgumentParser(description="Модель категорий письма")
    ap.add_argument("--data", default="data/processed/train.csv")
    ap.add_argument("--model-dir", default="artifacts")
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    build(df, Path(args.model_dir))


if __name__ == "__main__":
    main()
