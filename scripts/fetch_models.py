# -*- coding: utf-8 -*-
"""
fetch_models.py — скачивание обученных моделей из GitHub Releases.

Модель распознавания спама весит 23 МБ, поэтому она разбита на три части
(ограничение на размер одного запроса загрузки). Скрипт скачивает части,
склеивает их и проверяет, что модель загружается.

Используется при сборке Docker-образа на хостинге, а также локально:
    python scripts/fetch_models.py artifacts
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/Andrey123654/antispam-ml/releases/download/v1.0-models"
SPAM_PARTS = ["spam_bundle.part1", "spam_bundle.part2", "spam_bundle.part3"]
CATEGORY = "category_bundle.joblib"


def download(url: str, dest: Path, attempts: int = 5) -> None:
    last = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            dest.write_bytes(data)
            if len(data) == 0:
                raise IOError("пустой ответ")
            return
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  попытка {i + 1} не удалась: {e}", flush=True)
    raise RuntimeError(f"не удалось скачать {url}: {last}")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
    out.mkdir(parents=True, exist_ok=True)

    print("скачиваю части модели распознавания спама")
    chunks = []
    for name in SPAM_PARTS:
        tmp = out / f".{name}"
        download(f"{BASE}/{name}", tmp)
        chunks.append(tmp.read_bytes())
        print(f"  {name}: {len(chunks[-1]) / 1e6:.1f} МБ", flush=True)
        tmp.unlink(missing_ok=True)
    spam = out / "spam_bundle.joblib"
    spam.write_bytes(b"".join(chunks))
    print(f"  собрано: {spam.stat().st_size / 1e6:.1f} МБ")

    print("скачиваю модель категорий")
    download(f"{BASE}/{CATEGORY}", out / CATEGORY)

    # проверка загрузки
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import joblib  # noqa: E402
    bundle = joblib.load(spam)
    print("[ok] модели готовы:", sorted(p.name for p in out.glob("*.joblib")))
    print("     ключи модели:", sorted(bundle.keys()))


if __name__ == "__main__":
    main()
