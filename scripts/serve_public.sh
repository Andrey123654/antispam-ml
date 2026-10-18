#!/usr/bin/env bash
# Публичный демо-стенд: локальный сервис + туннель cloudflared (без регистрации).
# Печатает публичный https-URL, по которому сервис доступен из интернета.
# Стенд живёт, пока запущен этот скрипт.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8012}"
THRESHOLD="${SPAM_THRESHOLD:-0.80}"

command -v cloudflared >/dev/null 2>&1 || {
  echo "нет cloudflared. Установите: brew install cloudflared"
  echo "или скачайте: https://github.com/cloudflare/cloudflared/releases"
  exit 1
}

echo "== запускаю сервис на порту $PORT (порог $THRESHOLD) =="
SPAM_THRESHOLD="$THRESHOLD" .venv/bin/uvicorn service.app:app --host 127.0.0.1 --port "$PORT" &
SERVICE_PID=$!
trap 'kill $SERVICE_PID 2>/dev/null || true' EXIT
sleep 8

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && echo "== сервис отвечает =="

echo "== поднимаю публичный туннель =="
cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate 2>&1 | tee /tmp/tunnel.log &
TUNNEL_PID=$!

for _ in $(seq 1 40); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/tunnel.log | head -1 || true)
  [ -n "${URL:-}" ] && break
  sleep 1
done

if [ -n "${URL:-}" ]; then
  echo
  echo "==============================================================="
  echo "  ПУБЛИЧНЫЙ АДРЕС СЕРВИСА: $URL"
  echo "  документация:            $URL/docs"
  echo "  проверка:                curl $URL/health"
  echo "==============================================================="
else
  echo "не удалось получить публичный URL, смотрите /tmp/tunnel.log"
fi

wait $TUNNEL_PID
