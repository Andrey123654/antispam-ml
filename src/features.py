# -*- coding: utf-8 -*-
"""
features.py — признаки письма для антиспам-модели.

Два блока:

1. **Текстовые** — TF-IDF по словам (1–2-граммы) и по символьным n-граммам (3–5).
   Символьные n-граммы устойчивы к обфускации («в*и*а*г*р*а», «frее» с латинской «е»)
   и к языковым особенностям, поэтому используются вместе со словами.

2. **Числовые** — 57 признаков, совместимых с корпусом SpamBase (частоты слов и
   символов, статистика заглавных букв), плюс метаданные письма (тема, отправитель,
   HTML, ссылки, «денежные» и «срочные» маркеры).

Совместимость со SpamBase важна: она позволяет обучить модель на 4 601 письме
SpamBase и применить её к любым письмам, у которых есть сырой текст.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- чистка текста

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
NUM_RE = re.compile(r"\d+")
WS_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]{1,300}>")
MULTI_PUNCT_RE = re.compile(r"([!?$*])\1{2,}")


def clean_for_tfidf(text: str) -> str:
    """Нормализация для TF-IDF: ссылки и числа заменяются токенами."""
    s = str(text).lower()
    s = TAG_RE.sub(" ", s)
    s = URL_RE.sub(" urltoken ", s)
    s = EMAIL_RE.sub(" emailtoken ", s)
    s = NUM_RE.sub(" numtoken ", s)
    s = MULTI_PUNCT_RE.sub(r"\1\1", s)
    s = WS_RE.sub(" ", s)
    return s.strip()


# ------------------------------------------------------- 57 признаков SpamBase

SPAMBASE_WORDS = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet", "order",
    "mail", "receive", "will", "people", "report", "addresses", "free", "business",
    "email", "you", "credit", "your", "font", "000", "money", "hp", "hpl", "george",
    "650", "lab", "labs", "telnet", "857", "data", "415", "85", "technology",
    "1999", "parts", "pm", "direct", "cs", "meeting", "original", "project", "re",
    "edu", "table", "conference",
]
SPAMBASE_CHARS = [";", "(", "[", "!", "$", "#"]
SPAMBASE_COLUMNS = (
    [f"word_freq_{w}" for w in SPAMBASE_WORDS]
    + [f"char_freq_{c}" for c in ["semicolon", "paren", "bracket", "exclam", "dollar", "hash"]]
    + ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
)

WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9$#]+")
CAPS_RUN_RE = re.compile(r"[A-Z]{1,}")

# маркеры, полезные для антиспама (реклама, деньги, срочность, подписка)
MARKERS = {
    "unsubscribe": r"unsubscrib|отписат|отписк",
    "free": r"\bfree\b|\bбесплатн",
    "win": r"\bwin\b|\bвыигр|\bприз",
    "cash": r"\bcash\b|\bmoney\b|\bденьг|\b\$[\d.,]+",
    "urgent": r"urgent|срочно|today only|act now|последний шанс",
    "credit": r"\bcredit card\b|\bvisa\b|\bmastercard\b|\bкарт[аеу]\b",
    "meds": r"viagra|cialis|pharmacy|таблет|лекарств",
    "meeting": r"\bmeeting\b|\bвстреч|\bсозвон",
    "job": r"\bvacancy\b|\bвакансия|\bresume\b|\bрезюме|\bsalary\b|\bзарплат",
    "dating": r"\bdating\b|\bзнакомств|\bодинок|\bflirt",
    "delivery": r"\bdelivery\b|\bдоставка|\btracking\b|\bтрек-номер",
    "security": r"\bverify\b|\bподтверд|\bpassword\b|\bпарол|\baccount.*suspend",
}


def _caps_stats(text: str) -> tuple[float, int, int]:
    runs = [len(m.group(0)) for m in CAPS_RUN_RE.finditer(text)]
    total = sum(runs)
    longest = max(runs) if runs else 0
    avg = float(np.mean(runs)) if runs else 0.0
    return avg, longest, total


def spambase_features(text: str) -> dict:
    """57 признаков в том же смысле, что и корпус SpamBase (частоты в процентах)."""
    s = str(text)
    words = WORD_RE.findall(s.lower())
    n_words = max(len(words), 1)
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    f: dict[str, float] = {}
    for w in SPAMBASE_WORDS:
        f[f"word_freq_{w}"] = 100.0 * counts.get(w, 0) / n_words
    n_chars = max(len(s), 1)
    for col, ch in zip(["semicolon", "paren", "bracket", "exclam", "dollar", "hash"],
                       SPAMBASE_CHARS):
        f[f"char_freq_{col}"] = 100.0 * s.count(ch) / n_chars
    avg, longest, total = _caps_stats(s)
    f["capital_run_length_average"] = avg
    f["capital_run_length_longest"] = float(longest)
    f["capital_run_length_total"] = float(total)
    return f


# ------------------------------------------------------- метаданные письма

def meta_features(row: pd.Series) -> dict:
    """Признаки темы, отправителя, структуры и лексики письма."""
    subject = str(row.get("subject", "") or "")
    body = str(row.get("body", "") or "")
    text = subject + "\n" + body
    low = text.lower()
    n_words = max(len(WORD_RE.findall(low)), 1)

    urls = URL_RE.findall(low)
    domains = set()
    for u in urls:
        try:
            host = urlsplit(u if "://" in u else "http://" + u).netloc.lower()
            domains.add(host.partition(":")[0])
        except Exception:  # noqa: BLE001
            pass

    sender = str(row.get("sender", "") or "")
    sender_domain = ""
    if "@" in sender:
        sender_domain = sender.rsplit("@", 1)[-1].strip(" >").lower()

    f = {
        "subj_len": len(subject),
        "subj_words": len(WORD_RE.findall(subject.lower())),
        "body_len": len(body),
        "body_words": n_words,
        "uniq_word_ratio": len(set(WORD_RE.findall(low))) / n_words,
        "upper_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "digit_ratio": sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        "punct_ratio": sum(1 for c in text if c in "!?$%*#") / max(len(text), 1),
        "n_urls": float(len(urls)),
        "n_domains": float(len(domains)),
        "has_https": float(any("https://" in u for u in urls)),
        "url_per_1k": 1000.0 * len(urls) / max(len(text), 1),
        "is_html": float(row.get("is_html", 0) or 0),
        "has_attach": float(row.get("has_attach", 0) or 0),
        "n_parts": float(row.get("n_parts", 0) or 0),
        "subj_is_reply": float(bool(re.match(r"\s*(re|fwd|fw)\s*:", subject, re.I))),
        "subj_all_caps": float(subject.isupper() and len(subject) > 3),
        "subj_exclam": float(subject.count("!")),
        "sender_len": float(len(sender)),
        "sender_digits": float(sum(c.isdigit() for c in sender_domain)),
        "sender_freemail": float(bool(re.search(
            r"(gmail|yahoo|hotmail|mail\.ru|yandex|outlook|aol)\.", sender_domain))),
        "sender_hyphen": float("-" in sender_domain),
        "text_len_log": float(np.log1p(len(text))),
        "cyrillic_ratio": sum(1 for c in text if "\u0400" <= c <= "\u04ff") / max(len(text), 1),
    }
    for name, pat in MARKERS.items():
        f[f"m_{name}"] = float(bool(re.search(pat, low)))
    f["m_total"] = float(sum(f[f"m_{n}"] for n in MARKERS))
    return f


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Числовая матрица: признаки SpamBase + метаданные."""
    rows = []
    for _, row in df.iterrows():
        rec = spambase_features(row.get("text", ""))
        rec.update(meta_features(row))
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


NUMERIC_COLUMNS = SPAMBASE_COLUMNS + [
    "subj_len", "subj_words", "body_len", "body_words", "uniq_word_ratio",
    "upper_ratio", "digit_ratio", "punct_ratio", "n_urls", "n_domains",
    "has_https", "url_per_1k", "is_html", "has_attach", "n_parts",
    "subj_is_reply", "subj_all_caps", "subj_exclam", "sender_len",
    "sender_digits", "sender_freemail", "sender_hyphen", "text_len_log", "cyrillic_ratio",
] + [f"m_{n}" for n in MARKERS] + ["m_total"]
