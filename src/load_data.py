# -*- coding: utf-8 -*-
"""
load_data.py — сборка единого датасета из трёх независимых источников.

Источники:
  * SpamAssassin public corpus  — письма в формате RFC822 (ham / spam, включая hard_ham);
  * Ling-Spam                   — письма лингвистического списка рассылки (ham / spam);
  * SpamBase (UCI)              — 4 601 письмо, представленное 57 числовыми признаками
                                  (частоты слов и символов, статистика заглавных букв).

Первые два источника дают сырой текст письма, третий — табличные признаки.
Из SpamBase берутся только числовые признаки: они воспроизводимы на сырых письмах
(см. features.py), что позволяет использовать модель, обученную на SpamBase,
для любых писем.

Запуск:
    python src/load_data.py --raw data/raw --out data/processed/dataset.csv
"""

from __future__ import annotations

import argparse
import email
import html
import os
import re
import sys
from email import policy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAG_RE = re.compile(r"<[^>]{0,400}>", re.S)
SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
WS_RE = re.compile(r"[ \t\u00a0]+")
NL_RE = re.compile(r"\n{2,}")


def html_to_text(raw: str) -> str:
    """Грубое, но устойчивое извлечение текста из HTML-письма."""
    s = SCRIPT_RE.sub(" ", raw)
    s = TAG_RE.sub(" ", s)
    s = html.unescape(s)
    return WS_RE.sub(" ", s)


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = WS_RE.sub(" ", s)
    s = NL_RE.sub("\n", s)
    return s.strip()


def parse_message(raw: bytes) -> dict:
    """Извлекает тему, отправителя, текстовую часть и служебные признаки письма."""
    out = {"subject": "", "sender": "", "body": "", "is_html": 0, "has_attach": 0,
           "n_parts": 0, "raw_len": len(raw)}
    try:
        msg = email.message_from_bytes(raw, policy=policy.default)
    except Exception:  # noqa: BLE001
        try:
            msg = email.message_from_bytes(raw, policy=policy.compat32)
        except Exception:  # noqa: BLE001
            out["body"] = raw.decode("utf-8", "ignore")[:20000]
            return out

    def h(name: str) -> str:
        try:
            v = msg.get(name)
            return str(v) if v else ""
        except Exception:  # noqa: BLE001
            return ""

    out["subject"] = clean_text(h("Subject"))[:400]
    out["sender"] = h("From")[:200]

    plain, html_parts = [], []
    try:
        parts = list(msg.walk())
    except Exception:  # noqa: BLE001
        parts = [msg]
    out["n_parts"] = sum(1 for p in parts if p.get_content_maintype() != "multipart")
    for part in parts:
        ctype = part.get_content_type()
        if ctype == "text/plain":
            try:
                plain.append(part.get_content())
            except Exception:  # noqa: BLE001
                payload = part.get_payload(decode=True) or b""
                plain.append(payload.decode("utf-8", "ignore"))
        elif ctype == "text/html":
            try:
                html_parts.append(part.get_content())
            except Exception:  # noqa: BLE001
                payload = part.get_payload(decode=True) or b""
                html_parts.append(payload.decode("utf-8", "ignore"))
        elif part.get_content_maintype() == "multipart":
            continue
        else:
            out["has_attach"] = 1

    text = clean_text("\n".join(plain)) if plain else ""
    if html_parts:
        out["is_html"] = 1
        html_text = clean_text(html_to_text("\n".join(html_parts)))
        # если plain-части нет или она короткая, берём текст из HTML
        text = text if len(text) >= len(html_text) * 0.6 and text else html_text
    if not text:
        try:
            payload = msg.get_payload(decode=True) or b""
        except Exception:  # noqa: BLE001
            payload = b""
        text = clean_text(payload.decode("utf-8", "ignore"))
    out["body"] = text[:30000]
    return out


def _read_message_file(path: Path) -> dict:
    try:
        raw = path.read_bytes()
    except Exception:  # noqa: BLE001
        raw = b""
    return parse_message(raw)


def load_spamassassin(root: Path) -> pd.DataFrame:
    """SpamAssassin: каталоги ham (easy_ham, easy_ham_2, hard_ham) и spam (spam, spam_2)."""
    rows = []
    mapping = {"easy_ham": 0, "easy_ham_2": 0, "hard_ham": 0, "spam": 1, "spam_2": 1}
    sa_root = root / "sa"
    for dirname, label in mapping.items():
        d = sa_root / dirname
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.name.startswith("."):
                continue
            rec = _read_message_file(f)
            rec.update({"label": label, "source": f"spamassassin:{dirname}",
                        "orig_id": f.name})
            rows.append(rec)
    return pd.DataFrame(rows)


def load_lingspam(root: Path) -> pd.DataFrame:
    """
    Ling-Spam: письма лежат файлами прямо в частях bare/part*.
    Спам помечен префиксом имени файла: spmsg*.txt.
    """
    rows = []
    base = root / "lingspam_public" / "bare"
    if not base.is_dir():
        return pd.DataFrame(rows)
    for part in sorted(base.iterdir()):
        if not part.is_dir():
            continue
        for f in sorted(part.iterdir()):
            if not f.is_file() or f.suffix.lower() != ".txt":
                continue
            is_spam = f.name.lower().startswith("spmsg")
            rec = _read_message_file(f)
            rec.update({"label": 1 if is_spam else 0,
                        "source": f"lingspam:{part.name}", "orig_id": f.name})
            rows.append(rec)
    return pd.DataFrame(rows)


def load_test(root: Path) -> pd.DataFrame:
    """Тестовый набор из задания: каталоги ham/ и spam/."""
    rows = []
    for sub, label in (("ham", 0), ("spam", 1)):
        d = root / sub
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file():
                continue
            rec = _read_message_file(f)
            rec.update({"label": label, "source": "test", "orig_id": f.name})
            rows.append(rec)
    return pd.DataFrame(rows)


def load_spambase(root: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    SpamBase: 4 601 письмо × 57 числовых признаков + метка.
    Возвращает таблицу признаков и список имён признаков (из .names по порядку).
    """
    path = root / "spambase" / "spambase.data"
    df = pd.read_csv(path, header=None)
    names = spambase_feature_names(root)
    df.columns = names + ["label"]
    return df, names


def spambase_feature_names(root: Path) -> list[str]:
    """Имена 57 признаков SpamBase в порядке столбцов."""
    return SPAMBASE_COLUMNS


# Порядок столбцов SpamBase (word_freq_*, char_freq_*, capital_run_length_*)
SPAMBASE_COLUMNS = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses",
    "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you",
    "word_freq_credit", "word_freq_your", "word_freq_font", "word_freq_000",
    "word_freq_money", "word_freq_hp", "word_freq_hpl", "word_freq_george",
    "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet",
    "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
    "word_freq_technology", "word_freq_1999", "word_freq_parts",
    "word_freq_pm", "word_freq_direct", "word_freq_cs", "word_freq_meeting",
    "word_freq_original", "word_freq_project", "word_freq_re", "word_freq_edu",
    "word_freq_table", "word_freq_conference",
    "char_freq_semicolon", "char_freq_paren", "char_freq_bracket",
    "char_freq_exclam", "char_freq_dollar", "char_freq_hash",
    "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total",
]


def build(raw: Path, out: Path, with_test: bool = True) -> pd.DataFrame:
    frames = []
    sa = load_spamassassin(raw)
    if len(sa):
        print(f"  SpamAssassin: {len(sa)} писем (спам {int(sa['label'].mean() * 100)} %)")
        frames.append(sa)
    ls = load_lingspam(raw)
    if len(ls):
        print(f"  Ling-Spam:    {len(ls)} писем (спам {int(ls['label'].mean() * 100)} %)")
        frames.append(ls)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # чистка: убираем пустые письма и точные дубликаты текста
    before = len(df)
    df["body"] = df["body"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)
    df = df[(df["body"].str.len() > 30) | (df["subject"].str.len() > 10)]
    df["text"] = (df["subject"] + "\n" + df["body"]).str.strip()
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    print(f"  после чистки: {len(df)} писем (удалено {before - len(df)})")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[ok] обучающая выборка {out}: {df.shape}")
    print(f"  источники: {df['source'].str.split(':').str[0].value_counts().to_dict()}")
    print(f"  классы: {df['label'].value_counts().to_dict()}")

    if with_test:
        te = load_test(raw)
        if len(te):
            te["body"] = te["body"].fillna("").astype(str)
            te["subject"] = te["subject"].fillna("").astype(str)
            te["text"] = (te["subject"] + "\n" + te["body"]).str.strip()
            te = te.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
            overlap = len(set(df["text"]) & set(te["text"]))
            test_out = out.parent / "test.csv"
            te.to_csv(test_out, index=False)
            print(f"[ok] тестовая выборка {test_out}: {te.shape}")
            print(f"  классы: {te['label'].value_counts().to_dict()}")
            print(f"  пересечение текстов train/test: {overlap}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка единого датасета писем")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed/dataset.csv")
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()
    build(Path(args.raw), Path(args.out), with_test=not args.no_test)


if __name__ == "__main__":
    main()
