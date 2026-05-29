from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppConfig:
    """Параметры проекта, которые нужны обучению и сохранению артефактов."""

    data_path: Path
    model_path: Path
    metrics_path: Path
    test_size: float = 0.25
    random_state: int = 42
    max_features: int = 250
    regularization: float = 1.0
    epochs: int = 120


def _parse_scalar(value: str) -> Any:
    """Преобразует простое YAML-значение в int, float, bool или строку."""

    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def read_simple_yaml(path: Path) -> dict[str, Any]:
    """Читает плоский YAML-конфиг без внешней зависимости на PyYAML."""

    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return data


def load_config(config_path: str | Path = "configs/config.yaml") -> AppConfig:
    """Загружает конфиг и разрешает относительные пути от корня проекта."""

    path = Path(config_path)
    base_dir = path.parent.parent if path.parent.name == "configs" else Path.cwd()
    raw = read_simple_yaml(path)

    def resolve(value: str) -> Path:
        # В конфиге удобно хранить короткие относительные пути.
        candidate = Path(value)
        return candidate if candidate.is_absolute() else base_dir / candidate

    return AppConfig(
        data_path=resolve(str(raw["data_path"])),
        model_path=resolve(str(raw["model_path"])),
        metrics_path=resolve(str(raw["metrics_path"])),
        test_size=float(raw.get("test_size", 0.25)),
        random_state=int(raw.get("random_state", 42)),
        max_features=int(raw.get("max_features", 250)),
        regularization=float(raw.get("regularization", 1.0)),
        epochs=int(raw.get("epochs", 120)),
    )
