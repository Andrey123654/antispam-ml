"""Скачивание обученных моделей из GitHub Releases с проверкой контрольных сумм.

Манифест artifacts/model_manifest.json хранит путь, размер, SHA256 и адрес
каждого файла. Файлы скачиваются целиком.

Использование:
    python scripts/fetch_models.py [каталог]
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path, size: int, digest: str, attempts: int = 5) -> None:
    if dest.exists() and dest.stat().st_size == size and sha256(dest) == digest:
        print("проверено, скачивать не нужно:", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".download")
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "antispam-coursework/3.0"})
            with urllib.request.urlopen(request, timeout=600) as response:
                data = response.read()
            if len(data) != size:
                raise ValueError(f"размер {len(data)} вместо {size}")
            actual = hashlib.sha256(data).hexdigest()
            if actual != digest:
                raise ValueError(f"контрольная сумма {actual[:12]} не совпадает")
            tmp.write_bytes(data)
            tmp.replace(dest)
            print("скачано:", dest)
            return
        except Exception as error:  # noqa: BLE001
            last_error = error
            print(f"попытка {attempt} не удалась: {error}")
    raise SystemExit(f"не удалось скачать {url}: {last_error}")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts"
    manifest = json.loads((ROOT / "artifacts/model_manifest.json").read_text(encoding="utf-8"))
    for model in manifest["models"]:
        download(model["url"], out / model["path"], model["size"], model["sha256"])
    print("модели готовы:", ", ".join(m["path"] for m in manifest["models"]))


if __name__ == "__main__":
    main()
