"""Train the email adaptation with OOF meta training and a held-out threshold.

Run: python -m src.adapt_hackathon
The historical v1 model is preserved. Test labels never select the threshold.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train import BaseModels, build_sparse, clean_for_tfidf, logit, metrics_at
from hackathon_adapter import FEATURE_VERSION, numeric_with_hackathon

SEED = 42


def read_emails(path):
    df = pd.read_csv(path).fillna({"subject": "", "body": "", "sender": ""})
    df["text"] = df["subject"].astype(str) + "\n" + df["body"].astype(str)
    df["identity"] = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def base_fit(df, num):
    xs, wv, cv = build_sparse(df.text, fit=True)
    w = wv.transform(df.text.map(clean_for_tfidf))
    svd = TruncatedSVD(n_components=64, random_state=SEED).fit(w)
    dense = np.hstack([num.values, svd.transform(w)])
    return {"word_vec": wv, "char_vec": cv, "svd": svd,
            "base": BaseModels().fit(xs, dense, df.label.values)}


def base_predict(bundle, df, num):
    xs = build_sparse(df.text, bundle["word_vec"], bundle["char_vec"], fit=False)[0]
    w = bundle["word_vec"].transform(df.text.map(clean_for_tfidf))
    dense = np.hstack([num.values, bundle["svd"].transform(w)])
    return bundle["base"].predict_proba(xs, dense)


def predict_bundle(bundle, df):
    num = numeric_with_hackathon(df)[bundle["numeric_columns"]]
    return bundle["meta"].predict_proba(np.hstack([logit(base_predict(bundle, df, num)), num]))[:, 1]


def threshold_from_validation(y, scores, max_fpr=.01):
    choices = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
    best = None
    for t in choices:
        pred = scores >= t
        fp = int((pred & (y == 0)).sum()); tp = int((pred & (y == 1)).sum())
        if fp / max(int((y == 0).sum()), 1) > max_fpr:
            continue
        f1 = 2 * tp / max(int(pred.sum()) + int(y.sum()), 1)
        key = (tp, f1, -fp, float(t))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1]


def main():
    started = time.time()
    out = ROOT / "artifacts" / "hackathon_v2"; out.mkdir(exist_ok=True)
    pool = read_emails(ROOT / "data/processed/train.csv")
    extra = read_emails(ROOT / "data/processed/synthetic_ham.csv")
    pool = pd.concat([pool, extra], ignore_index=True)
    test = read_emails(ROOT / "data/processed/test.csv")
    conflicting = set(pool.groupby("identity").label.nunique().loc[lambda x: x > 1].index)
    before = len(pool)
    pool = pool[~pool.identity.isin(conflicting)].drop_duplicates("identity").reset_index(drop=True)
    original_identities = set(pool.identity)
    original_unseen = ~test.identity.isin(original_identities)
    overlap = int(test.identity.isin(original_identities).sum())
    # Purge every exact normalized test text from both development and validation.
    pool = pool[~pool.identity.isin(set(test.identity))].reset_index(drop=True)
    tr_idx, va_idx = train_test_split(np.arange(len(pool)), test_size=.2,
                                     stratify=pool.label, random_state=SEED)
    tr, va = pool.iloc[tr_idx].reset_index(drop=True), pool.iloc[va_idx].reset_index(drop=True)
    assert not set(tr.identity) & set(va.identity)
    assert not set(pool.identity) & set(test.identity)
    print(f"development={len(tr)}, validation={len(va)}, test={len(test)}, purged={overlap}", flush=True)
    num = numeric_with_hackathon(tr)
    print(f"numeric features={num.shape[1]}", flush=True)
    oof = np.zeros((len(tr), 4))
    for fold, (it, iv) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, tr.label), 1):
        b = base_fit(tr.iloc[it], num.iloc[it])
        oof[iv] = base_predict(b, tr.iloc[iv], num.iloc[iv])
        print(f"fold {fold}/5 finished", flush=True)
    # Meta model only sees base predictions made without the row being trained on.
    meta = make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=3000,
                         class_weight="balanced", random_state=SEED))
    meta.fit(np.hstack([logit(oof), num]), tr.label)
    bundle = base_fit(tr, num)
    bundle.update(kind=FEATURE_VERSION, meta=meta, numeric_columns=list(num.columns), seed=SEED)
    va_num = numeric_with_hackathon(va)
    va_base = base_predict(bundle, va, va_num)
    va_score = meta.predict_proba(np.hstack([logit(va_base), va_num]))[:, 1]
    thr = threshold_from_validation(va.label.values, va_score)
    bundle["threshold"] = thr
    validation = metrics_at(va.label.values, va_score, thr)
    baseline_thr = threshold_from_validation(va.label.values, va_base[:, 0])
    ptest = predict_bundle(bundle, test)
    report = {
        "schema_version": 2, "kind": FEATURE_VERSION,
        "basis": {"repository": "https://github.com/Andrey123654/Hakaton",
                  "commit": "afd2206d6d172b369b620d01ba10d6776219ad90",
                  "file": "src/features.py",
                  "sha256": hashlib.sha256((ROOT / "src/vendor/hackathon_features.py").read_bytes()).hexdigest(),
                  "transfer": "24 offline URL features, max/mean aggregation plus count/truncation; OOF stacking",
                  "not_transferred": "phishing weights, target labels, network snapshots, public F1"},
        "dataset": {"input_pool": before, "pool_after_test_purge": len(pool),
                    "development": len(tr), "validation": len(va), "test": len(test),
                    "normalized_test_overlap_purged": overlap,
                    "test_unseen_relative_to_legacy_pool": int(original_unseen.sum()),
                    "conflicting_pool_texts_removed": len(conflicting),
                    "sources_development": tr.source.str.split(":").str[0].value_counts().to_dict(),
                    "spambase_used_for_training": False},
        "threshold": thr,
        "threshold_policy": "maximum validation recall under empirical FPR <= 1%; no test tuning",
        "cv_base_models": {name: metrics_at(tr.label.values, oof[:, j], .5)
                           for j, name in enumerate(["lr", "svm", "nb", "gb"])},
        "validation": validation,
        "validation_text_lr_baseline": metrics_at(va.label.values, va_base[:, 0], baseline_thr),
        "test_full": metrics_at(test.label.values, ptest, thr),
        "test_unseen": metrics_at(test.loc[original_unseen, "label"].values, ptest[original_unseen], thr),
        "evaluation_notes": ["All test texts purged from development and validation before fitting.",
            "Full test and unseen subset use the exact same bundle and threshold.",
            "Validation is used for threshold selection; it is not an independent final score.",
            "Base model CV is OOF. No nested-CV claim is made for the final stacking model.",
            "Historical test has been inspected in prior work; fresh external validation remains necessary.",
            "Near duplicates and source/campaign grouping have not been eliminated."],
        "runtime_sec": round(time.time() - started, 1)}
    joblib.dump(bundle, out / "spam_bundle.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    pd.DataFrame({"label": test.label, "score": ptest, "predicted": (ptest >= thr).astype(int),
                  "legacy_unseen": original_unseen}).to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "development_oof.npz", base=oof, label=tr.label.values)
    split = pd.concat([tr.assign(split="development"), va.assign(split="validation"), test.assign(split="test")])
    pd.DataFrame({"sha256": [hashlib.sha256(s.encode()).hexdigest() for s in split.identity],
                  "label": split.label.values, "split": split.split.values}).to_csv(out / "split_manifest.csv", index=False)
    print(json.dumps({"validation": validation, "test": report["test_full"], "unseen": report["test_unseen"],
                      "runtime_sec": report["runtime_sec"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
