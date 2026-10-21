#!/usr/bin/env bash
# Скачивание публичных корпусов для обучения антиспам-модели.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/sa data/raw/spambase

echo "== SpamAssassin public corpus =="
for f in 20021010_easy_ham 20021010_hard_ham 20021010_spam \
         20030228_easy_ham 20030228_easy_ham_2 20030228_hard_ham \
         20030228_spam 20030228_spam_2 20050311_spam_2; do
  curl -sSL -o "data/raw/sa/${f}.tar.bz2" "https://spamassassin.apache.org/old/publiccorpus/${f}.tar.bz2"
  tar -xjf "data/raw/sa/${f}.tar.bz2" -C data/raw/sa
done

echo "== Ling-Spam =="
curl -sSL -o data/raw/lingspam.tar.gz "http://www.aueb.gr/users/ion/data/lingspam_public.tar.gz"
tar -xzf data/raw/lingspam.tar.gz -C data/raw

echo "== SpamBase (UCI) =="
curl -sSL -o data/raw/spambase.zip "https://archive.ics.uci.edu/static/public/94/spambase.zip"
unzip -o -q data/raw/spambase.zip -d data/raw/spambase

echo "== Тестовый набор из задания (Яндекс.Диск) =="
python - <<'PY'
import json, urllib.parse, urllib.request
key = "https://disk.yandex.ru/d/mJ_9zG3dX3eZgg"
u = ("https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
     + urllib.parse.quote(key, safe=""))
href = json.loads(urllib.request.urlopen(u).read())["href"]
urllib.request.urlretrieve(href, "data/test_task_emails.tar.xz")
PY
mkdir -p data/raw && tar -xJf data/test_task_emails.tar.xz -C data/raw

echo "готово: данные в data/raw/"
