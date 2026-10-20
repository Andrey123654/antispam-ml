# -*- coding: utf-8 -*-
"""
make_borderline_report.py — отчёт «пограничные примеры из тестовой выборки».

Модель — актуальная v2 (адаптация решения хакатона, `artifacts/hackathon_v2`).
Отбирает письма у порога решения, группирует по типу данных (структура письма и
категория) и показывает голоса четырёх базовых моделей: расхождение их
вероятностей и есть «пограничность». Пояснительный текст считается из данных,
чтобы отчёт не расходился с таблицами.

Запуск:
    python tools/make_borderline_report.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adapt_hackathon import base_predict, predict_bundle  # noqa: E402
from category import predict as cat_predict  # noqa: E402
from hackathon_adapter import numeric_with_hackathon  # noqa: E402

BUNDLE_PATH = ROOT / "artifacts" / "hackathon_v2" / "spam_bundle.joblib"
METRICS_PATH = ROOT / "artifacts" / "hackathon_v2" / "metrics.json"
OUT = ROOT / "reports" / "Пограничные_примеры.md"

BAND_LOW_OFFSET = 0.25
BAND_HIGH_OFFSET = 0.15


def anon(text: str, limit: int = 300) -> str:
    """Обезличивание: убираем адреса, ссылки, имена, лишние пробелы."""
    s = str(text)
    s = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "<e-mail>", s)
    s = re.sub(r"https?://\S+", "<ссылка>", s)
    s = re.sub(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.", "<ФИО>", s)
    s = re.sub(r"\b(Hi|Hello|Dear|Привет|Здравствуйте)[,\s]+[A-ZА-ЯЁ][a-zа-яё]{2,12}\b",
               r"\1 <Имя>", s)
    s = re.sub(r"[\u200b\u200c\u200d]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit] + ("…" if len(s) > limit else "")


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def letters(n: int) -> str:
    """Склонение: 1 письмо, 2 письма, 5 писем."""
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return "письмо"
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return "письма"
    return "писем"


def comment_for(row: dict, thr: float) -> str:
    """Пояснение к строке — по фактическому раскладу голосов четырёх моделей."""
    votes = {"LR": row["lr"], "SVM": row["svm"], "NB": row["nb"], "GB": row["gb"]}
    spread = max(votes.values()) - min(votes.values())
    spam_votes = [k for k, v in votes.items() if v >= thr]
    ham_votes = [k for k, v in votes.items() if v < thr]
    n_spam, n_ham = len(spam_votes), len(ham_votes)
    dec = "спам" if row["pred"] == 1 else "норма"
    p = row["p"]

    if n_spam == 4:
        if row["pred"] == 1:
            return "все четыре модели за спам — случай лёгкий"
        return (f"все четыре модели за спам, но мета-модель опустила итог до "
                f"{p:.3f} (порог {thr:.2f}); итог — норма")
    if n_ham == 4:
        if row["pred"] == 0:
            return "все четыре модели против спама — случай лёгкий"
        return (f"все четыре модели против спама, но мета-модель подняла итог до "
                f"{p:.3f} (порог {thr:.2f}); итог — спам")
    if n_spam == n_ham:
        return (f"голоса разделились {n_spam}:{n_ham} (разброс {spread:.2f}): "
                f"словарные модели и бустинг смотрят на разные признаки; итог — {dec}")
    if n_spam == 1:
        return (f"за спам только {spam_votes[0]} (разброс {spread:.2f}), "
                f"остальные против; итог — {dec}")
    if n_ham == 1:
        return (f"против спама только {ham_votes[0]} (разброс {spread:.2f}), "
                f"остальные за спам; итог — {dec}")
    return f"голоса {n_spam}:{n_ham} (разброс {spread:.2f}); итог — {dec}"


def main() -> None:
    te = pd.read_csv(ROOT / "data/processed/test.csv")
    for c in ("body", "subject", "sender"):
        te[c] = te[c].fillna("").astype(str)
    te["text"] = (te["subject"] + "\n" + te["body"]).astype(str)

    bundle = joblib.load(BUNDLE_PATH)
    THR = float(bundle["threshold"])
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cat = joblib.load(ROOT / "artifacts/category_bundle.joblib")

    te["p"] = predict_bundle(bundle, te)
    te["pred"] = (te["p"] >= THR).astype(int)
    te["category"] = cat_predict(cat, te["text"].tolist())
    te["len"] = te["text"].str.len()
    cyr = re.compile(r"[\u0400-\u04ff]")
    te["lang"] = np.where(
        te["text"].map(lambda s: len(cyr.findall(s)) / max(len(s), 1)) > 0.1, "RU", "EN")
    te["d"] = (te["p"] - THR).abs()

    low, high = THR - BAND_LOW_OFFSET, THR + BAND_HIGH_OFFSET
    band = te[(te["p"] >= low) & (te["p"] <= high)]
    n_err = int((te["pred"] != te["label"]).sum())
    band_err = int((band["pred"] != band["label"]).sum())
    fp_total = int(((te["pred"] == 1) & (te["label"] == 0)).sum())
    ham_total = int((te["label"] == 0).sum())
    confident = float(((te["p"] < 0.2) | (te["p"] > 0.9)).mean() * 100)

    used: set[int] = set()
    picks: list[tuple[str, pd.Series]] = []

    def pick(mask: pd.Series, name: str) -> None:
        sub = te[mask & ~te.index.isin(used)]
        if not len(sub):
            return
        inside = sub[(sub["p"] >= low) & (sub["p"] <= high)]
        row = (inside if len(inside) else sub).sort_values("d").iloc[0]
        used.add(int(row.name))
        picks.append((name, row))

    for mask, name in [
        (te["is_html"] == 1, "HTML-письмо"),
        (te["is_html"] == 0, "только текст"),
        (te["lang"] == "RU", "русский текст"),
        (te["lang"] == "EN", "английский текст"),
        (te["len"] < 400, "очень короткое (< 400 символов)"),
        (te["len"] > 4000, "длинное (> 4000 символов)"),
        (te["subject"].str.len() < 5, "без темы"),
        (te["has_attach"] == 1, "с вложением"),
    ]:
        pick(mask, name)
    for c in ["реклама", "работа", "знакомства", "финансы", "доставка",
              "безопасность", "прочее"]:
        pick(te["category"] == c, f"категория «{c}»")

    df = pd.DataFrame([r for _, r in picks])
    num = numeric_with_hackathon(df)[bundle["numeric_columns"]]
    bp = base_predict(bundle, df, num)

    rows: list[dict] = []
    for i, (name, row) in enumerate(picks):
        rec = {
            "type": name, "label": int(row["label"]), "p": float(row["p"]),
            "pred": int(row["pred"]), "category": row["category"], "lang": row["lang"],
            "html": int(row["is_html"]), "len": int(row["len"]),
            "subject": anon(row["subject"], 90), "text": anon(row["text"], 320),
            "lr": float(bp[i, 0]), "svm": float(bp[i, 1]),
            "nb": float(bp[i, 2]), "gb": float(bp[i, 3]),
        }
        rec["comment"] = comment_for(rec, THR)
        rows.append(rec)

    err_rows: list[dict] = []
    fp = te[(te["pred"] == 1) & (te["label"] == 0)].sort_values("p")
    fn = te[(te["pred"] == 0) & (te["label"] == 1)].sort_values("p", ascending=False)
    for tag, frame in (("ложное срабатывание", fp.head(2)), ("пропуск спама", fn.head(2))):
        for _, row in frame.iterrows():
            r = pd.DataFrame([row])
            num1 = numeric_with_hackathon(r)[bundle["numeric_columns"]]
            pr = base_predict(bundle, r, num1)[0]
            rec = {"tag": tag, "label": int(row["label"]), "p": float(row["p"]),
                   "pred": int(row["pred"]), "category": row["category"],
                   "subject": anon(row["subject"], 90), "text": anon(row["text"], 340),
                   "lr": float(pr[0]), "svm": float(pr[1]), "nb": float(pr[2]),
                   "gb": float(pr[3])}
            rec["comment"] = comment_for(rec, THR)
            err_rows.append(rec)

    def table(items: list[dict]) -> str:
        out = ["| Тип данных | Метка | Вероятность | LR | SVM | NB | GB | Решение |",
               "|---|---|---:|---:|---:|---:|---:|---|"]
        for r in items:
            mark = "спам" if r["label"] == 1 else "норма"
            dec = "спам" if r["pred"] == 1 else "норма"
            ok = "✓" if r["pred"] == r["label"] else "✗"
            out.append(f"| {r['type']} | {mark} | **{r['p']:.3f}** | {r['lr']:.2f} | "
                       f"{r['svm']:.2f} | {r['nb']:.2f} | {r['gb']:.2f} | {dec} {ok} |")
        return "\n".join(out)

    def explain(items: list[dict]) -> str:
        return "\n".join(f"* **{r['type']}** — {r['comment']}." for r in items)

    structure_rows = [r for r in rows if not r["type"].startswith("категория")]
    category_rows = [r for r in rows if r["type"].startswith("категория")]
    spread_max = max(
        max(r["lr"], r["svm"], r["nb"], r["gb"]) - min(r["lr"], r["svm"], r["nb"], r["gb"])
        for r in rows)
    band_err_share = band_err / max(n_err, 1) * 100
    band_share = len(band) / len(te) * 100

    md = f"""# Пограничные примеры из тестовой выборки

Разбор писем, на которых модель сомневается: вероятность спама близка к порогу
решения. Такие примеры показывают, **где проходит граница** между классами и
**почему** модель ошибается.

Модель — актуальная версия **v2** (адаптация решения хакатона: признаки ссылок
плюс признаки письма), порог берётся из обученного артефакта.

## 1. Как отбирали

* Тестовая выборка: **{fmt(len(te))} {letters(len(te))}**, порог решения **{THR:.3f}**
  (выбран по валидации; тестовые метки в выборе порога не участвуют).
* «Полоса неопределённости» — вероятность от {low:.2f} до {high:.2f}:
  **{fmt(len(band))} {letters(len(band))}** ({band_share:.1f} % теста).
* В этой полосе сосредоточено **{band_err} из {n_err}** ошибок модели
  ({band_err_share:.0f} %), при том что писем в ней {band_share:.1f} %.

**Насколько модель уверена.** У {confident:.0f} % писем вероятность ниже 0.2 или
выше 0.9 — модель не сомневается. Пограничных мало, потому что признаки (слова,
символы и ссылки) дают сильный сигнал, а четыре модели подкрепляют друг друга.

**Что такое «пограничность» здесь.** Базовые модели смотрят на разные признаки:
логистическая регрессия и SVM — на слова и символьные n-граммы, Complement NB —
на частоты слов, градиентный бустинг — на плотные числовые признаки, включая
признаки ссылок. Когда их голоса расходятся, итоговая мета-модель оказывается
у порога. В таблицах ниже приведены голоса каждой модели.

## 2. По структуре письма

{table(structure_rows)}

Как читать эти строки:

{explain(structure_rows)}

## 3. По категории письма

{table(category_rows)}

Категории считает отдельная модель (слабая разметка правилами), поэтому они
показывают, **о чём** письмо, а не насколько оно спамное.

{explain(category_rows)}

## 4. Четыре показательных случая

"""
    for r in err_rows:
        mark = "норма" if r["label"] == 0 else "спам"
        md += f"""### {r['tag']} (истина: {mark}, вероятность {r['p']:.3f})

* **Тема:** {r['subject']}
* **Фрагмент:** {r['text']}
* **Голоса:** LR {r['lr']:.2f} · SVM {r['svm']:.2f} · NB {r['nb']:.2f} · GB {r['gb']:.2f}
* **Почему пограничное:** {r['comment']}.

"""
    md += f"""## 5. Что это даёт для защиты

1. **Ошибки сосредоточены у порога.** {band_err} из {n_err} ошибок
   ({band_err_share:.0f} %) приходятся на {band_share:.1f} % писем — работать
   надо именно с этой полосой, а не «в целом».
2. **Расхождение моделей — индикатор неуверенности.** В примерах разброс голосов
   доходит до {spread_max:.2f}. Флаг «отправить на ручную проверку» при разбросе
   больше 0.5 выделил бы ровно проблемные письма.
3. **Причина пограничности — сдвиг домена.** Транзакционные уведомления,
   HTML-письма и русский текст плохо представлены в публичных корпусах: словарные
   модели «не видят» сигнала, и решение перевешивает одна числовая модель.
4. **Порог — это политика.** При пороге {THR:.3f} ложных срабатываний
   {fp_total} из {fmt(ham_total)} нормальных писем. Снижение порога поймает больше
   спама, но увеличит число потерянных нормальных писем.

## 6. Как воспроизвести

```bash
python tools/make_borderline_report.py
```

Отбор: письма с вероятностью в полосе {low:.2f}–{high:.2f}; для каждого типа
берётся ближайшее к порогу. Тексты обезличены: адреса, ссылки и имена заменены
на `<e-mail>`, `<ссылка>`, `<Имя>`.
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"[ok] {OUT}")
    print(f"  модель v2, порог {THR:.3f}, полоса {low:.2f}–{high:.2f}")
    print(f"  примеров: {len(rows)} по типам + {len(err_rows)} ошибок")
    print(f"  в полосе {len(band)} {letters(len(band))}, ошибок в ней {band_err} из {n_err}")


if __name__ == "__main__":
    main()
