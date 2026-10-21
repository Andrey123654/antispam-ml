#!/usr/bin/env bash
# Полный цикл: данные → обучение → категории.
set -euo pipefail
cd "$(dirname "$0")"

[ -d data/raw/sa ] || bash scripts/download_data.sh
[ -f data/processed/train.csv ] || .venv/bin/python src/load_data.py --raw data/raw --out data/processed/train.csv
.venv/bin/python -m src.adapt_hackathon
[ -f artifacts/category_bundle.joblib ] || .venv/bin/python -m src.category
echo "готово: модель и метрики в artifacts/hackathon_v2/"
