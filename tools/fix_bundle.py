# -*- coding: utf-8 -*-
"""
fix_bundle.py — пересохранение артефакта модели с корректной ссылкой на класс.

Если модель обучалась запуском `python src/train.py`, класс BaseModels попадает
в pickle как `__main__.BaseModels` и не находится при загрузке из сервиса.
Скрипт читает бандл с подстановкой и сохраняет заново так, чтобы класс
ссылался на модуль `train`.

Запуск:
    python tools/fix_bundle.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import train  # noqa: E402


def main() -> None:
    main_mod = sys.modules.get("__main__") or types.ModuleType("__main__")
    main_mod.BaseModels = train.BaseModels
    sys.modules["__main__"] = main_mod

    path = ROOT / "artifacts" / "spam_bundle.joblib"
    bundle = joblib.load(path)

    train.BaseModels.__module__ = "train"
    joblib.dump(bundle, path, compress=3)

    check = joblib.load(path)
    print("[ok] артефакт пересохранён:", path)
    print("  размер:", round(path.stat().st_size / 1e6, 1), "МБ")
    print("  ключи:", sorted(check.keys()))
    print("  класс базовых моделей:", type(check["base"]).__name__,
          "из модуля", type(check["base"]).__module__)


if __name__ == "__main__":
    main()
