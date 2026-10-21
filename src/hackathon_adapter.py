"""Offline transfer of URL features from the author's phishing hackathon.

This module reuses feature extraction, not phishing labels or the tournament
F1. No DNS, HTTP, TLS or RDAP requests are made for submitted emails.
"""
from functools import lru_cache
import re
import numpy as np
import pandas as pd
from features import numeric_frame
from vendor.hackathon_features import extract_features

FEATURE_VERSION = "hackathon-email-v2"
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
URL_KEYS = ["url_len", "host_len", "path_len", "n_subdomains", "n_hyphens_host",
            "digits_ratio_host", "has_userinfo", "is_ip_host", "is_punycode",
            "has_hex_encoding", "entropy_host", "path_depth", "suspicious_tld",
            "is_free_host", "is_shortener", "n_phish_words", "brand_mimicry",
            "brand_embedded_in_sld", "is_typosquatting", "is_homoglyph_attack",
            "has_login_word", "has_prize_word", "ext_is_exec", "n_attack_techniques"]
MAX_URLS = 50


def urls_in(text):
    return [u.rstrip(".,;:!?)]}")[:2048] for u in URL_RE.findall(str(text))]


@lru_cache(maxsize=30000)
def url_values(url):
    try:
        f = extract_features(url)
        return tuple(float(f[k]) for k in URL_KEYS)
    except (ValueError, UnicodeError):
        return (0.0,) * len(URL_KEYS)


def email_url_features(text):
    urls = urls_in(text)
    a = np.asarray([url_values(u) for u in urls[:MAX_URLS]], dtype=float)
    out = {"hack_url_count": len(urls), "hack_urls_truncated": int(len(urls) > MAX_URLS)}
    for j, key in enumerate(URL_KEYS):
        out[f"hack_max_{key}"] = float(a[:, j].max()) if len(a) else 0.0
        out[f"hack_mean_{key}"] = float(a[:, j].mean()) if len(a) else 0.0
    return out


def numeric_with_hackathon(df):
    base = numeric_frame(df).reset_index(drop=True)
    url = pd.DataFrame([email_url_features(t) for t in df["text"]])
    return pd.concat([base, url], axis=1).replace([np.inf, -np.inf], 0).fillna(0)
