#!/usr/bin/env bash
# Скачивает обученные модели из раздела Releases на GitHub и собирает их обратно.
# Нужен, потому что модель весит 23 МБ и разбита на три части (ограничение загрузки).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts

BASE="https://github.com/Andrey123654/antispam-ml/releases/download/v1.0-models"

echo "== скачиваю части модели =="
for i in 1 2 3; do
  curl -sSL --retry 5 --retry-all-errors -o "/tmp/spam_bundle.part$i" "$BASE/spam_bundle.part$i"
  echo "  часть $i: $(du -h /tmp/spam_bundle.part$i | cut -f1)"
done
cat /tmp/spam_bundle.part1 /tmp/spam_bundle.part2 /tmp/spam_bundle.part3 > artifacts/spam_bundle.joblib
rm -f /tmp/spam_bundle.part1 /tmp/spam_bundle.part2 /tmp/spam_bundle.part3

echo "== скачиваю модель категорий =="
curl -sSL --retry 5 --retry-all-errors -o artifacts/category_bundle.joblib "$BASE/category_bundle.joblib"

echo "== проверка =="
ls -lh artifacts/*.joblib
.venv/bin/python - <<'PY' 2>/dev/null || python - <<'PY'
import sys, joblib
sys.path.insert(0, "src")
from train import predict_bundle
import pandas as pd
b = joblib.load("artifacts/spam_bundle.joblib")
df = pd.DataFrame([{"text": "Congratulations! You won $1,000,000. Click here!",
                    "subject": "WINNER", "sender": "a@b.biz", "body": "",
                    "is_html": 0, "has_attach": 0, "n_parts": 1}])
print("модель загружена, вероятность спама:", round(float(predict_bundle(b, df)[0]), 4))
PY
