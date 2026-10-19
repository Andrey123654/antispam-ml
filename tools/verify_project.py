"""Validate the saved adaptation, split isolation and the real API contract."""
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from service.app import app, _to_frame, Email
from adapt_hackathon import predict_bundle, read_emails
from hackathon_adapter import email_url_features


def main():
    folder = ROOT / "artifacts/hackathon_v2"
    m = json.loads((folder / "metrics.json").read_text())
    assert hashlib.sha256((ROOT / "src/vendor/hackathon_features.py").read_bytes()).hexdigest() == m["basis"]["sha256"]
    split = pd.read_csv(folder / "split_manifest.csv")
    sets = {name: set(part.sha256) for name, part in split.groupby("split")}
    assert not sets["development"] & sets["validation"]
    assert not (sets["development"] | sets["validation"]) & sets["test"]
    predictions = pd.read_csv(folder / "test_predictions.csv")
    full = m["test_full"]; unseen = m["test_unseen"]
    assert unseen["fp"] <= full["fp"] and unseen["tp"] <= full["tp"]
    assert np.array_equal(predictions.predicted, predictions.score >= m["threshold"])
    assert full["fp"] == int(((predictions.label == 0) & (predictions.predicted == 1)).sum())
    assert email_url_features("Meeting tomorrow")["hack_url_count"] == 0
    assert email_url_features("https://trusted.example@evil.example/login")["hack_max_has_userinfo"] == 1
    bundle = joblib.load(folder / "spam_bundle.joblib")
    test = read_emails(ROOT / "data/processed/test.csv")
    examples = [Email(text=str(test.iloc[i].body), subject=str(test.iloc[i].subject),
                      sender=str(test.iloc[i].sender)) for i in [0, 1, 10, 30]]
    expected = predict_bundle(bundle, _to_frame(examples))
    with TestClient(app) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["threshold"] == m["threshold"]
        assert not h["threshold_overridden"]
        response = client.post("/predict_batch", json=[e.model_dump() for e in examples])
        assert response.status_code == 200
        assert np.allclose([x["spam_probability"] for x in response.json()], expected, atol=5.1e-6)
        assert client.post("/predict", json={"text": ""}).status_code == 422
        assert client.post("/predict_batch", json=[]).status_code == 400
        assert client.post("/predict_batch", json=[{"text": "a"}] * 257).status_code == 400
        assert "Проверка письма" in client.get("/").text
    result = {"status": "passed", "checks": ["source checksum", "split isolation", "same-model subset evaluation",
              "threshold parity", "offline URL features", "saved-model/API parity", "request bounds", "demo page"],
              "test_rows": len(predictions), "threshold": m["threshold"]}
    (ROOT / "reports/validation/checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
