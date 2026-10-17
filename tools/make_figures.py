# -*- coding: utf-8 -*-
"""
make_figures.py — рисунки для презентации и отчёта.

Готовит:
  * схему пайплайна (источники → признаки → базовые модели → стекинг → сервис);
  * столбиковую диаграмму качества базовых моделей и стекинга;
  * матрицу ошибок на кросс-валидации;
  * распределение вероятностей спама (спам против нормальных);
  * «скриншоты» терминала с ответами сервиса.

Запуск:
    python tools/make_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FIG = ROOT / "reports" / "figures"
SCREEN = ROOT / "reports" / "screens"
FIG.mkdir(parents=True, exist_ok=True)
SCREEN.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.family": "DejaVu Sans",
})
BLUE, RED, GREY, GREEN = "#2E5BFF", "#E5484D", "#8B8D98", "#12A594"


def pipeline_scheme(metrics: dict) -> None:
    """Схема полного цикла: данные → признаки → модели → стекинг → сервис."""
    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, lines, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    linewidth=1.6, edgecolor=color, facecolor=color + "14"))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top",
                fontsize=11, fontweight="bold", color=color)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.62 - i * 0.3, line, ha="center", va="top",
                    fontsize=8.6, color="#22262E")

    box(0.15, 1.25, 2.1, 2.6, "Данные", [
        "SpamAssassin · 6 155", "Ling-Spam · 2 875",
        "SpamBase · 4 601", "синтетика · 25",
        "тест · 1 545 писем"], BLUE)
    box(2.65, 1.25, 2.1, 2.6, "Признаки", [
        "TF-IDF слова 1–2", "60 000 признаков",
        "TF-IDF символы 3–4", "200 000 признаков",
        "94 числовых признака"], GREEN)
    box(5.15, 1.25, 2.25, 2.6, "Базовые модели", [
        "логистическая регрессия", "линейный SVM",
        "Complement NB", "градиентный бустинг"], "#8E4EC6")
    box(7.8, 1.25, 2.1, 2.6, "Стекинг", [
        "логиты 4 моделей", "+ 94 признака",
        "логистическая регрессия", "порог по кросс-валидации"], "#D6409F")
    box(10.3, 1.25, 2.15, 2.6, "Сервис", [
        "FastAPI", "Docker", "/predict", "/predict_batch", "/health"], "#F76B15")

    for x0, x1 in [(2.25, 2.65), (4.75, 5.15), (7.4, 7.8), (9.9, 10.3)]:
        ax.add_patch(FancyArrowPatch((x0 + 0.02, 2.55), (x1 - 0.02, 2.55),
                                     arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.8, color="#22262E"))
    ax.text(6.3, 4.15, "Полный цикл: сбор данных → признаки → модели → упаковка → публикация",
            ha="center", fontsize=12.5, fontweight="bold", color="#111318")
    cv = metrics["cv"]["stack"]
    ax.text(6.3, 0.75, f"Качество: F1 = {cv['f1']:.3f}, precision = {cv['precision']:.3f}, "
                       f"recall = {cv['recall']:.3f} на кросс-валидации   ·   "
                       f"ложных срабатываний {cv['false_positive_rate'] * 100:.1f} %",
            ha="center", fontsize=10, color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "pipeline.png", bbox_inches="tight")
    plt.close(fig)


def quality_bars(metrics: dict) -> None:
    """F1 / Precision / Recall базовых моделей и стекинга на кросс-валидации."""
    base = metrics["cv"]["base_models"]
    stack = metrics["cv"]["stack"]
    names = {"lr": "Логистическая\nрегрессия", "svm": "Линейный\nSVM",
             "nb": "Complement\nNB", "gb": "Градиентный\nбустинг"}
    labels = [names.get(k, k) for k in base] + ["Стекинг"]
    f1 = [base[k]["f1"] for k in base] + [stack["f1"]]
    prec = [base[k]["precision"] for k in base] + [stack["precision"]]
    rec = [base[k]["recall"] for k in base] + [stack["recall"]]

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.bar(x - w, f1, w, label="F1", color=BLUE)
    ax.bar(x, prec, w, label="Precision", color=GREEN)
    ax.bar(x + w, rec, w, label="Recall", color="#F76B15")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("значение метрики")
    ax.set_title("Качество базовых моделей и стекинга (кросс-валидация, 9 055 писем)",
                 fontsize=12, fontweight="bold", pad=26)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              fontsize=9)
    for i, v in enumerate(f1):
        ax.text(i - w, v + 0.002, f"{v:.3f}", ha="center", fontsize=7.6)
    fig.tight_layout()
    fig.savefig(FIG / "quality.png", bbox_inches="tight")
    plt.close(fig)


def confusion_plot(metrics: dict) -> None:
    """Матрица ошибок на кросс-валидации."""
    m = metrics["cv"]["stack"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["нормальное", "спам"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("предсказание")
    ax.set_ylabel("истина")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", " "), ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "#111318")
    ax.set_title("Матрица ошибок, кросс-валидация", fontsize=12, fontweight="bold")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "confusion.png", bbox_inches="tight")
    plt.close(fig)


def probability_hist(model_path: Path, test: Path) -> None:
    """Распределение вероятностей спама на тестовом наборе."""
    import joblib
    from train import predict_bundle
    bundle = joblib.load(model_path)
    df = pd.read_csv(test)
    df["body"] = df["body"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["text"] = (df["subject"] + "\n" + df["body"]).astype(str)
    p = predict_bundle(bundle, df)
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    bins = np.linspace(0, 1, 40)
    ax.hist(p[df["label"] == 0], bins=bins, alpha=0.75, label="нормальные", color=GREEN)
    ax.hist(p[df["label"] == 1], bins=bins, alpha=0.65, label="спам", color=RED)
    ax.set_yscale("log")
    ax.set_xlabel("вероятность спама")
    ax.set_ylabel("число писем (лог. шкала)")
    ax.set_title("Разделение классов на тестовом наборе", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "probability.png", bbox_inches="tight")
    plt.close(fig)


def terminal_screenshot(lines: list[str], out: Path, title: str = "Терминал") -> None:
    """Псевдо-скриншот терминала: тёмный фон, моноширинный текст."""
    font = None
    for f in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(f):
            try:
                font = ImageFont.truetype(f, 13)
                break
            except Exception:  # noqa: BLE001
                continue
    if font is None:
        font = ImageFont.load_default()
    pad, lh = 16, 18
    width = max(560, int(max(len(l) for l in lines) * 7.6) + pad * 2)
    height = lh * len(lines) + pad * 2 + 26
    img = Image.new("RGB", (width, height), (24, 26, 32))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, 24], fill=(38, 41, 50))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([12 + i * 18, 7, 22 + i * 18, 17], fill=c)
    d.text((width - 130, 5), title, font=font, fill=(180, 184, 194))
    for i, line in enumerate(lines):
        color = (120, 220, 140) if line.strip().startswith(("$", ">")) else (214, 218, 226)
        d.text((pad, 34 + i * lh), line, font=font, fill=color)
    img.save(out)


def main() -> None:
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
    pipeline_scheme(metrics)
    quality_bars(metrics)
    confusion_plot(metrics)
    try:
        probability_hist(ROOT / "artifacts" / "spam_bundle.joblib",
                         ROOT / "data" / "processed" / "test.csv")
    except Exception as e:  # noqa: BLE001
        print("распределение вероятностей не построено:", e)
    print("[ok] рисунки:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
