from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .features import (
    apply_scaling,
    build_vocabulary,
    readability_features,
    scale_features,
    vectorize,
)

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
DEFAULT_CEFR_THRESHOLDS = [0.10, 0.30, 0.50, 0.70, 0.90]
DEFAULT_LEVEL_CENTERS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


@dataclass
class TextComplexityModel:
    """Линейная модель сложности текста с отдельно калиброванными CEFR-порогами."""

    vocabulary: list[str]
    feature_names: list[str]
    means: dict[str, dict[str, float]]
    coefficients: dict[str, float]
    intercept: float
    cefr_thresholds: list[float] = field(default_factory=lambda: DEFAULT_CEFR_THRESHOLDS.copy())

    def predict_score(self, text: str) -> float:
        """Возвращает непрерывный score сложности в диапазоне от 0 до 1."""

        raw = vectorize(text, self.vocabulary)
        features = apply_scaling(raw, self.means)
        learned_score = self.intercept + sum(
            self.coefficients.get(name, 0.0) * features.get(name, 0.0)
            for name in self.feature_names
        )
        return min(1.0, max(0.0, learned_score))

    def cefr_level_for_score(self, score: float) -> str:
        return score_to_cefr_level(score, self.cefr_thresholds)

    def level_for_score(self, score: float) -> str:
        return score_to_level(score, self.cefr_thresholds)

    def confidence_for_score(self, score: float) -> float:
        return confidence_from_score(score, self.cefr_thresholds)

    def predict(self, text: str) -> dict[str, object]:
        """Собирает полный ответ API: score, уровень, уверенность и интерпретируемые признаки."""

        score = self.predict_score(text)
        return {
            "score": round(score, 4),
            "level": self.level_for_score(score),
            "cefr_level": self.cefr_level_for_score(score),
            "confidence": round(self.confidence_for_score(score), 4),
            "features": readability_features(text),
        }

    def save(self, path: str | Path) -> None:
        """Сохраняет модель и пороги в JSON-артефакт."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "vocabulary": self.vocabulary,
                    "feature_names": self.feature_names,
                    "means": self.means,
                    "coefficients": self.coefficients,
                    "intercept": self.intercept,
                    "cefr_thresholds": self.cefr_thresholds,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TextComplexityModel":
        """Восстанавливает модель из JSON-артефакта."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            vocabulary=list(data["vocabulary"]),
            feature_names=list(data["feature_names"]),
            means=data["means"],
            coefficients={key: float(value) for key, value in data["coefficients"].items()},
            intercept=float(data["intercept"]),
            cefr_thresholds=_normalize_thresholds(
                [float(value) for value in data.get("cefr_thresholds", DEFAULT_CEFR_THRESHOLDS)]
            ),
        )


def score_to_level(score: float, thresholds: list[float] | None = None) -> str:
    """Сворачивает CEFR-шкалу в три укрупнённые категории сложности."""

    cefr_level = score_to_cefr_level(score, thresholds)
    if cefr_level in {"A1", "A2"}:
        return "easy"
    if cefr_level in {"B1", "B2"}:
        return "medium"
    return "hard"


def score_to_cefr_level(score: float, thresholds: list[float] | None = None) -> str:
    """Переводит числовой score в CEFR-уровень с учётом калиброванных порогов."""

    boundaries = _normalize_thresholds(thresholds or DEFAULT_CEFR_THRESHOLDS)
    score = round(min(1.0, max(0.0, score)), 2)
    if score < boundaries[0]:
        return "A1"
    if score < boundaries[1]:
        return "A2"
    if score < boundaries[2]:
        return "B1"
    if score < boundaries[3]:
        return "B2"
    if score < boundaries[4]:
        return "C1"
    return "C2"


def confidence_from_score(score: float, thresholds: list[float] | None = None) -> float:
    """Оценивает уверенность по расстоянию до ближайшей границы уровня."""

    boundaries = _normalize_thresholds(thresholds or DEFAULT_CEFR_THRESHOLDS)
    distance = min(abs(score - boundary) for boundary in boundaries)
    return min(0.95, 0.55 + distance * 1.5)


def calibrate_cefr_thresholds(
    model: TextComplexityModel,
    texts: list[str],
    levels: list[str],
) -> list[float]:
    """Подбирает пороги между уровнями по валидационному распределению score."""

    if len(texts) != len(levels):
        raise ValueError("texts и levels должны иметь одинаковую длину")
    if not texts:
        return DEFAULT_CEFR_THRESHOLDS.copy()

    # Для каждого уровня собираем score модели, чтобы затем поставить границы
    # не вручную, а между реальными кластерами предсказаний.
    grouped_scores: dict[str, list[float]] = {level: [] for level in CEFR_LEVELS}
    for text, level in zip(texts, levels):
        if level not in grouped_scores:
            continue
        grouped_scores[level].append(model.predict_score(text))

    centers: list[float] = []
    for default_center, level in zip(DEFAULT_LEVEL_CENTERS, CEFR_LEVELS):
        scores = sorted(grouped_scores[level])
        if not scores:
            # Если в валидации не встретился уровень, оставляем разумный центр по умолчанию.
            centers.append(default_center)
            continue
        middle = len(scores) // 2
        if len(scores) % 2:
            centers.append(scores[middle])
        else:
            centers.append((scores[middle - 1] + scores[middle]) / 2.0)

    centers = _normalize_centers(centers)
    thresholds = [
        (left + right) / 2.0
        for left, right in zip(centers, centers[1:])
    ]
    return _normalize_thresholds(thresholds)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _normalize_centers(values: list[float], min_gap: float = 0.04) -> list[float]:
    """Упорядочивает центры уровней и не даёт им схлопнуться друг с другом."""

    if not values:
        raise ValueError("Список values не должен быть пустым")
    adjusted = [_clamp(float(value)) for value in values]
    lower_limit = 0.0
    for index, value in enumerate(adjusted):
        remaining = len(adjusted) - index - 1
        upper_limit = 1.0 - remaining * min_gap
        adjusted[index] = min(upper_limit, max(lower_limit, value))
        lower_limit = adjusted[index] + min_gap
    return adjusted


def _normalize_thresholds(thresholds: list[float], min_gap: float = 0.04) -> list[float]:
    """Проверяет и выравнивает список из пяти CEFR-порогов."""

    if len(thresholds) != 5:
        raise ValueError("Требуется ровно пять CEFR-порогов")
    adjusted: list[float] = []
    lower_limit = 0.01
    for index, value in enumerate(thresholds):
        remaining = len(thresholds) - index - 1
        upper_limit = 0.99 - remaining * min_gap
        adjusted_value = min(upper_limit, max(lower_limit, float(value)))
        adjusted.append(round(adjusted_value, 4))
        lower_limit = adjusted_value + min_gap
    return adjusted


def _macro_f1(true_levels: list[str], predicted_levels: list[str]) -> float:
    """Считает macro F1 без внешних библиотек."""

    scores: list[float] = []
    for level in CEFR_LEVELS:
        true_positive = sum(
            1
            for true, predicted in zip(true_levels, predicted_levels)
            if true == level and predicted == level
        )
        false_positive = sum(
            1
            for true, predicted in zip(true_levels, predicted_levels)
            if true != level and predicted == level
        )
        false_negative = sum(
            1
            for true, predicted in zip(true_levels, predicted_levels)
            if true == level and predicted != level
        )
        if true_positive == 0 and false_positive == 0 and false_negative == 0:
            continue
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_cefr_classification(true_levels: list[str], predicted_levels: list[str]) -> dict[str, float]:
    """Возвращает основные метрики CEFR-классификации."""

    if len(true_levels) != len(predicted_levels):
        raise ValueError("true_levels и predicted_levels должны иметь одинаковую длину")
    level_to_index = {level: index for index, level in enumerate(CEFR_LEVELS)}
    accuracy = sum(
        true == predicted for true, predicted in zip(true_levels, predicted_levels)
    ) / len(true_levels)
    adjacent_accuracy = sum(
        abs(level_to_index[true] - level_to_index[predicted]) <= 1
        for true, predicted in zip(true_levels, predicted_levels)
        if true in level_to_index and predicted in level_to_index
    ) / len(true_levels)
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(_macro_f1(true_levels, predicted_levels), 4),
        "adjacent_accuracy": round(adjacent_accuracy, 4),
    }


def _ranks(values: list[float]) -> list[float]:
    """Строит ранги с усреднением на совпадающих значениях."""

    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + end + 1) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def spearman_correlation(true_values: list[float], predicted_values: list[float]) -> float:
    """Считает корреляцию Спирмена без зависимости на scipy."""

    if len(true_values) != len(predicted_values):
        raise ValueError("true_values и predicted_values должны иметь одинаковую длину")
    if not true_values:
        return 0.0
    true_ranks = _ranks(true_values)
    predicted_ranks = _ranks(predicted_values)
    mean_true = sum(true_ranks) / len(true_ranks)
    mean_predicted = sum(predicted_ranks) / len(predicted_ranks)
    numerator = sum(
        (true_rank - mean_true) * (predicted_rank - mean_predicted)
        for true_rank, predicted_rank in zip(true_ranks, predicted_ranks)
    )
    true_denominator = math.sqrt(sum((rank - mean_true) ** 2 for rank in true_ranks))
    predicted_denominator = math.sqrt(
        sum((rank - mean_predicted) ** 2 for rank in predicted_ranks)
    )
    if not true_denominator or not predicted_denominator:
        return 0.0
    return numerator / (true_denominator * predicted_denominator)


def train_ridge_regression(
    texts: list[str],
    targets: list[float],
    max_features: int,
    regularization: float,
    epochs: int = 1200,
    learning_rate: float = 0.03,
) -> TextComplexityModel:
    """Обучает простую ridge-регрессию градиентным спуском."""

    vocabulary = build_vocabulary(texts, max_features=max_features)
    rows = [vectorize(text, vocabulary) for text in texts]
    scaled_rows, means = scale_features(rows)
    feature_names = sorted(means.keys())
    coefficients = {name: 0.0 for name in feature_names}
    intercept = sum(targets) / len(targets)
    n_rows = len(scaled_rows)

    # Для проекта достаточно прозрачной реализации оптимизации:
    # это позволяет легко читать веса, сериализовать модель и обходиться без sklearn.
    for _ in range(epochs):
        grad_b = 0.0
        grad_w = {name: 0.0 for name in feature_names}
        for row, target in zip(scaled_rows, targets):
            prediction = intercept + sum(coefficients[name] * row[name] for name in feature_names)
            error = prediction - target
            grad_b += error
            for name in feature_names:
                grad_w[name] += error * row[name]
        intercept -= learning_rate * grad_b / n_rows
        for name in feature_names:
            penalty = regularization * coefficients[name]
            coefficients[name] -= learning_rate * (grad_w[name] / n_rows + penalty / n_rows)

    return TextComplexityModel(
        vocabulary=vocabulary,
        feature_names=feature_names,
        means=means,
        coefficients=coefficients,
        intercept=intercept,
    )


def evaluate(
    model: TextComplexityModel,
    texts: list[str],
    targets: list[float],
    levels: list[str] | None = None,
) -> dict[str, float]:
    """Считает регрессионные и, при наличии меток, классификационные метрики."""

    predictions = [model.predict_score(text) for text in texts]
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))
    mean_target = sum(targets) / len(targets)
    total = sum((target - mean_target) ** 2 for target in targets)
    residual = sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions))
    r2 = 1.0 - residual / total if total else 0.0
    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "spearman": round(spearman_correlation(targets, predictions), 4),
    }
    if levels:
        predicted_levels = [model.cefr_level_for_score(prediction) for prediction in predictions]
        metrics.update(evaluate_cefr_classification(levels, predicted_levels))
    return metrics


def evaluate_constant_prediction(value: float, targets: list[float]) -> dict[str, float]:
    """Базовый baseline: модель всегда предсказывает одно и то же значение."""

    errors = [value - target for target in targets]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4)}
