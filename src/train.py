# -*- coding: utf-8 -*-
"""
train.py — обучение антиспам-модели.

Схема:
  1. Признаки: TF-IDF по словам (1–2-граммы) и по символьным n-граммам (3–5),
     плюс 57 признаков, совместимых со SpamBase, и метаданные письма.
  2. Базовые модели: логистическая регрессия, линейный SVM (калиброванный),
     Complement Naive Bayes, градиентный бустинг на плотных признаках.
  3. Мета-модель: логистическая регрессия на логитах базовых моделей и
     числовых признаках.
  4. Порог выбирается по OOF так, чтобы максимизировать F1 при низком числе
     ложноположительных срабатываний (это ключевое требование задачи:
     не отправлять в спам нормальные письма).

Оценка:
  * 5-фолдовая кросс-валидация на обучающей выборке (OOF-метрики);
  * тестовый набор из задания — целиком и отдельно на письмах, которых не было
    в обучении (в наборе есть точные копии писем публичных корпусов);
  * кросс-корпусная проверка: обучение на SpamAssassin → тест на Ling-Spam и наоборот.

Запуск:
    python src/train.py --train data/processed/train.csv --test data/processed/test.csv \
        --model-dir artifacts --report artifacts/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import NUMERIC_COLUMNS, clean_for_tfidf, numeric_frame  # noqa: E402

SEED = 42
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def build_sparse(texts: pd.Series, word_vec=None, char_vec=None, fit: bool = True):
    """TF-IDF по словам и символьным n-граммам (матрица разреженная)."""
    cleaned = texts.map(clean_for_tfidf)
    if fit:
        word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                   strip_accents="unicode", max_features=60000,
                                   token_pattern=r"(?u)\b\w+\b")
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=5,
                                   sublinear_tf=True, strip_accents="unicode",
                                   max_features=200000)
        word_vec.fit(cleaned)
        char_vec.fit(cleaned)
    Xw = word_vec.transform(cleaned)
    Xc = char_vec.transform(cleaned)
    return sparse.hstack([Xw, Xc]).tocsr(), word_vec, char_vec


def binarize(p: np.ndarray, thr: float) -> np.ndarray:
    return (p >= thr).astype(int)


def tune_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, dict]:
    """Порог по OOF: максимум F1, при равенстве — меньше ложноположительных."""
    best = (-1.0, 1e9, 0.5)
    for thr in np.unique(np.quantile(p, np.linspace(0.02, 0.98, 400))):
        pred = binarize(p, thr)
        f1 = f1_score(y, pred)
        fp = int(((pred == 1) & (y == 0)).sum())
        if f1 > best[0] + 1e-12 or (abs(f1 - best[0]) <= 1e-12 and fp < best[1]):
            best = (f1, fp, float(thr))
    return best[2], {"f1": best[0], "fp": best[1]}


def metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = binarize(p, thr)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(thr), 6),
        "f1": round(float(f1_score(y, pred)), 5),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 5),
        "roc_auc": round(float(roc_auc_score(y, p)), 5) if len(set(y)) > 1 else None,
        "pr_auc": round(float(average_precision_score(y, p)), 5) if len(set(y)) > 1 else None,
        "n": int(len(y)),
        "spam": int(y.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "false_positive_rate": round(float(fp / max(fp + tn, 1)), 5),
    }


class BaseModels:
    """Четыре базовые модели: две линейные на TF-IDF, NB и бустинг на плотных признаках."""

    def fit(self, Xs, Xd, y):
        self.lr = LogisticRegression(C=4.0, max_iter=3000, solver="liblinear",
                                     class_weight="balanced", random_state=SEED)
        self.lr.fit(Xs, y)
        self.svm = CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", random_state=SEED), cv=3, method="sigmoid")
        self.svm.fit(Xs, y)
        self.nb = ComplementNB(alpha=0.3)
        self.nb.fit(Xs, y)
        self.gb = HistGradientBoostingClassifier(
            max_iter=350, learning_rate=0.07, max_leaf_nodes=24,
            l2_regularization=1.0, random_state=SEED)
        self.gb.fit(Xd, y)
        return self

    def predict_proba(self, Xs, Xd) -> np.ndarray:
        cols = [self.lr.predict_proba(Xs)[:, 1],
                self.svm.predict_proba(Xs)[:, 1],
                self.nb.predict_proba(Xs)[:, 1],
                self.gb.predict_proba(Xd)[:, 1]]
        return np.column_stack(cols)


def fit_dense(texts, num, word_vec=None, char_vec=None, svd=None, fit=True):
    """Плотная матрица: числовые признаки + SVD от TF-IDF."""
    Xs, word_vec, char_vec = build_sparse(texts, word_vec, char_vec, fit)
    if fit:
        svd = TruncatedSVD(n_components=64, random_state=SEED, algorithm="randomized")
        svd.fit(word_vec.transform(cleaned))  # SVD только по словам
    dense = np.hstack([num.values.astype(np.float32),
                       svd.transform(word_vec.transform(cleaned)).astype(np.float32)])
    return dense, Xs, word_vec, char_vec, svd


def run_cv(df: pd.DataFrame, n_splits: int = 5) -> tuple[dict, np.ndarray, dict]:
    """Кросс-валидация: OOF-предсказание ансамбля и метрики базовых моделей."""
    y = df["label"].values
    num = numeric_frame(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_base = np.zeros((len(df), 4))
    for fold, (itr, iva) in enumerate(skf.split(df, y), 1):
        tr, va = df.iloc[itr], df.iloc[iva]
        Xs_tr, wv, cv_ = build_sparse(tr["text"], fit=True)
        Xs_va = build_sparse(va["text"], wv, cv_, fit=False)[0]
        W_tr = wv.transform(tr["text"].map(clean_for_tfidf))
        W_va = wv.transform(va["text"].map(clean_for_tfidf))
        svd = TruncatedSVD(n_components=64, random_state=SEED, algorithm="randomized").fit(W_tr)
        Xd_tr = np.hstack([num.iloc[itr].values, svd.transform(W_tr)])
        Xd_va = np.hstack([num.iloc[iva].values, svd.transform(W_va)])
        m = BaseModels().fit(Xs_tr, Xd_tr, y[itr])
        oof_base[iva] = m.predict_proba(Xs_va, Xd_va)
        print(f"  фолд {fold}: база обучена", flush=True)
    names = ["lr", "svm", "nb", "gb"]
    per_model = {n: metrics_at(y, oof_base[:, i], 0.5) for i, n in enumerate(names)}
    return per_model, oof_base, {"y": y, "num": num}


def stack_meta(oof_base: np.ndarray, num: pd.DataFrame, y: np.ndarray,
               n_splits: int = 5):
    """Мета-модель на логитах базовых моделей и числовых признаках (OOF)."""
    X = np.hstack([logit(oof_base), StandardScaler().fit_transform(num.values)])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for itr, iva in skf.split(X, y):
        meta = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")
        meta.fit(X[itr], y[itr])
        oof[iva] = meta.predict_proba(X[iva])[:, 1]
    return oof


def fit_final(df: pd.DataFrame) -> dict:
    """Обучение финального ансамбля на всех данных."""
    y = df["label"].values
    num = numeric_frame(df)
    Xs, wv, cv_ = build_sparse(df["text"], fit=True)
    W = wv.transform(df["text"].map(clean_for_tfidf))
    svd = TruncatedSVD(n_components=64, random_state=SEED, algorithm="randomized").fit(W)
    Xd = np.hstack([num.values, svd.transform(W)])
    base = BaseModels().fit(Xs, Xd, y)
    base_proba = base.predict_proba(Xs, Xd)
    scaler = StandardScaler().fit(num.values)
    Xm = np.hstack([logit(base_proba), scaler.transform(num.values)])
    meta = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(Xm, y)
    return {"word_vec": wv, "char_vec": cv_, "svd": svd, "base": base,
            "meta": meta, "scaler": scaler, "numeric_columns": NUMERIC_COLUMNS}


def predict_bundle(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Предсказание ансамбля для новых писем."""
    num = numeric_frame(df)
    Xs = build_sparse(df["text"], bundle["word_vec"], bundle["char_vec"], fit=False)[0]
    W = bundle["word_vec"].transform(df["text"].map(clean_for_tfidf))
    Xd = np.hstack([num.values, bundle["svd"].transform(W)])
    base_proba = bundle["base"].predict_proba(Xs, Xd)
    Xm = np.hstack([logit(base_proba), bundle["scaler"].transform(num.values)])
    return bundle["meta"].predict_proba(Xm)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение антиспам-модели")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--test", default="data/processed/test.csv")
    ap.add_argument("--model-dir", default="artifacts")
    ap.add_argument("--report", default="artifacts/metrics.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--extra", default="data/processed/synthetic_ham.csv",
                    help="дополнительные письма (транзакционные нормальные) в обучение")
    ap.add_argument("--skip-cross", action="store_true",
                    help="не выполнять кросс-корпусную проверку")
    args = ap.parse_args()

    t0 = time.time()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    train["body"] = train["body"].fillna("").astype(str)
    train["subject"] = train["subject"].fillna("").astype(str)
    train["text"] = (train["subject"] + "\n" + train["body"]).astype(str)
    extra_path = Path(args.extra)
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        extra["body"] = extra["body"].fillna("").astype(str)
        extra["subject"] = extra["subject"].fillna("").astype(str)
        extra["text"] = (extra["subject"] + "\n" + extra["body"]).astype(str)
        before = len(train)
        train = pd.concat([train, extra[train.columns.intersection(extra.columns)]],
                          ignore_index=True)
        print(f"добавлено транзакционных писем: {len(train) - before}")
    test["body"] = test["body"].fillna("").astype(str)
    test["subject"] = test["subject"].fillna("").astype(str)
    test["text"] = (test["subject"] + "\n" + test["body"]).astype(str)
    print(f"обучающая выборка: {len(train)} (спам {int(train['label'].mean()*100)} %), "
          f"тест: {len(test)} (спам {int(test['label'].mean()*100)} %)")

    # 1) кросс-валидация
    print("[1/4] кросс-валидация")
    per_model, oof_base, ctx = run_cv(train, args.folds)
    y = ctx["y"]
    for name, m in per_model.items():
        print(f"  {name:<4} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

    # 2) мета-модель
    print("[2/4] мета-модель")
    oof = stack_meta(oof_base, ctx["num"], y, args.folds)
    thr, _ = tune_threshold(y, oof)
    cv_metrics = metrics_at(y, oof, thr)
    print(f"  OOF F1={cv_metrics['f1']:.4f} P={cv_metrics['precision']:.4f} "
          f"R={cv_metrics['recall']:.4f} FP={cv_metrics['fp']} (порог {thr:.4f})")

    # 3) финальная модель и тест из задания
    print("[3/4] финальная модель и тестовый набор")
    bundle = fit_final(train)
    p_test = predict_bundle(bundle, test)
    test_metrics = metrics_at(test["label"].values, p_test, thr)

    # честная оценка: письма теста, которых не было в обучении
    seen = set(train["text"])
    unseen_mask = ~test["text"].isin(seen)
    unseen = test[unseen_mask].reset_index(drop=True)
    train_clean = train[~train["text"].isin(set(test["text"]))].reset_index(drop=True)
    bundle_clean = fit_final(train_clean)
    p_unseen = predict_bundle(bundle_clean, unseen)
    unseen_metrics = metrics_at(unseen["label"].values, p_unseen, thr)
    print(f"  тест целиком: F1={test_metrics['f1']:.4f} FP={test_metrics['fp']}")
    print(f"  только невиданные: n={unseen_metrics['n']} F1={unseen_metrics['f1']:.4f} "
          f"P={unseen_metrics['precision']:.4f} R={unseen_metrics['recall']:.4f}")

    # 4) кросс-корпусная проверка
    cross = {}
    if args.skip_cross:
        print("[4/4] кросс-корпусная проверка пропущена")
    else:
        print("[4/4] кросс-корпусная проверка")
    train["src"] = train["source"].str.split(":").str[0]
    for src_a, src_b in ([] if args.skip_cross else
                         [("spamassassin", "lingspam"), ("lingspam", "spamassassin")]):
        a = train[train["src"] == src_a].reset_index(drop=True)
        b = train[train["src"] == src_b].reset_index(drop=True)
        if len(a) < 200 or len(b) < 100:
            continue
        bundle_a = fit_final(a)
        p_b = predict_bundle(bundle_a, b)
        m = metrics_at(b["label"].values, p_b, thr)
        cross[f"{src_a}->{src_b}"] = m
        print(f"  {src_a} -> {src_b}: F1={m['f1']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f}")

    # сохранение
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, Path(args.model_dir) / "spam_bundle.joblib", compress=3)
    report = {
        "dataset": {
            "train": int(len(train)), "test": int(len(test)),
            "train_spam": int(y.sum()), "test_spam": int(test["label"].sum()),
            "sources": train["src"].value_counts().to_dict(),
            "test_overlap_with_train": int(len(test) - len(unseen)),
            "test_unseen": int(len(unseen)),
        },
        "cv": {"base_models": per_model, "stack": cv_metrics},
        "test_full": test_metrics,
        "test_unseen": unseen_metrics,
        "cross_corpus": cross,
        "threshold": round(float(thr), 6),
        "runtime_sec": round(time.time() - t0, 1),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"[ok] модель: {args.model_dir}/spam_bundle.joblib | отчёт: {args.report}")
    print(f"время: {report['runtime_sec']} с")


if __name__ == "__main__":
    main()
