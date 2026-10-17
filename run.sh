#!/usr/bin/env bash
# Полный цикл: данные → обучение → категории → сервис.
set -euo pipefail
cd "$(dirname "$0")"

[ -d data/raw/sa ] || bash scripts/download_data.sh
[ -f data/processed/train.csv ] || .venv/bin/python src/load_data.py --raw data/raw --out data/processed/train.csv
.venv/bin/python src/train.py
.venv/bin/python src/category.py
.venv/bin/python tools/make_figures.py
echo "готово: модель в artifacts/, метрики в artifacts/metrics.json"
