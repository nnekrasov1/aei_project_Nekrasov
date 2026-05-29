from __future__ import annotations

import csv
from dataclasses import dataclass
import random
from pathlib import Path

# CEFR-уровни переводятся в упорядоченную числовую шкалу,
# чтобы модель могла решать задачу как регрессию с последующей калибровкой порогов.
CEFR_TO_TARGET = {
    "A1": 0.0,
    "A2": 0.2,
    "B1": 0.4,
    "B2": 0.6,
    "C1": 0.8,
    "C2": 1.0,
}


@dataclass(frozen=True)
class TextSample:
    """Одна строка датасета в удобном для обучения виде."""

    text: str
    target: float
    level: str


def normalize_cefr_level(value: str) -> str:
    """Нормализует метку уровня и убирает варианты вида A1+."""

    level = value.strip().upper().replace("+", "")
    if level not in CEFR_TO_TARGET:
        raise ValueError(f"Неподдерживаемый CEFR-уровень: {value}")
    return level


def cefr_to_target(level: str) -> float:
    return CEFR_TO_TARGET[normalize_cefr_level(level)]


def load_dataset(path: str | Path) -> list[TextSample]:
    """Читает CSV и собирает список обучающих примеров."""

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            text = row.get("text", "").strip()
            if not text:
                continue
            cefr_level = row.get("cefr_level", "").strip()
            level = row.get("level", "").strip() or cefr_level
            # Если явный target не записан в CSV, восстанавливаем его из CEFR-уровня.
            target = (
                float(row["target"])
                if row.get("target")
                else cefr_to_target(cefr_level)
            )
            rows.append(TextSample(text=text, target=target, level=level))
    if not rows:
        raise ValueError(f"Датасет пуст: {path}")
    return rows


def train_test_split(
    samples: list[TextSample],
    test_size: float,
    random_state: int,
) -> tuple[list[TextSample], list[TextSample]]:
    """Делит выборку стратифицированно по уровням, чтобы классы не исчезали из holdout."""

    if not 0 < test_size < 1:
        raise ValueError("Параметр test_size должен быть между 0 и 1")
    grouped: dict[str, list[TextSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.level, []).append(sample)

    rng = random.Random(random_state)
    train_samples: list[TextSample] = []
    test_samples: list[TextSample] = []
    for level in sorted(grouped):
        level_samples = list(grouped[level])
        rng.shuffle(level_samples)
        # Для редких уровней оставляем хотя бы один пример в обучении,
        # иначе калибровка порогов и сама модель становятся нестабильнее.
        if len(level_samples) == 1:
            train_samples.extend(level_samples)
            continue
        test_count = max(1, round(len(level_samples) * test_size))
        test_count = min(test_count, len(level_samples) - 1)
        test_samples.extend(level_samples[:test_count])
        train_samples.extend(level_samples[test_count:])

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples
