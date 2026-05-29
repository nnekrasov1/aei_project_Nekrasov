from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import load_dataset, train_test_split
from .model import (
    calibrate_cefr_thresholds,
    evaluate,
    evaluate_constant_prediction,
    train_ridge_regression,
)


def train(config_path: str = "configs/config.yaml") -> dict[str, object]:
    """Полный цикл обучения: split, калибровка порогов, оценка и сохранение артефактов."""

    config = load_config(config_path)
    samples = load_dataset(config.data_path)

    # Первый split выделяет финальный holdout, который больше не участвует в настройке.
    train_pool, test_samples = train_test_split(
        samples,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    # Из train-пула отдельно выделяем validation только для калибровки CEFR-порогов.
    train_samples, validation_samples = train_test_split(
        train_pool,
        test_size=config.test_size,
        random_state=config.random_state + 1,
    )
    evaluated_model = train_ridge_regression(
        texts=[sample.text for sample in train_samples],
        targets=[sample.target for sample in train_samples],
        max_features=config.max_features,
        regularization=config.regularization,
        epochs=config.epochs,
    )
    # Границы между уровнями берём из валидации, а не из жёстко заданной таблицы.
    evaluated_model.cefr_thresholds = calibrate_cefr_thresholds(
        evaluated_model,
        texts=[sample.text for sample in validation_samples],
        levels=[sample.level for sample in validation_samples],
    )
    train_targets = [sample.target for sample in train_samples]
    test_targets = [sample.target for sample in test_samples]
    test_levels = [sample.level for sample in test_samples]
    baseline_value = sum(train_targets) / len(train_targets)
    baseline_metrics = evaluate_constant_prediction(baseline_value, test_targets)
    model_metrics = evaluate(
        evaluated_model,
        texts=[sample.text for sample in test_samples],
        targets=test_targets,
        levels=test_levels,
    )

    # Финальную модель переобучаем на полном train-пуле, но пороги оставляем
    # откалиброванными по отдельно отложенной validation-части.
    final_model = train_ridge_regression(
        texts=[sample.text for sample in train_pool],
        targets=[sample.target for sample in train_pool],
        max_features=config.max_features,
        regularization=config.regularization,
        epochs=config.epochs,
    )
    final_model.cefr_thresholds = calibrate_cefr_thresholds(
        final_model,
        texts=[sample.text for sample in validation_samples],
        levels=[sample.level for sample in validation_samples],
    )
    metrics = {
        "protocol": {
            "train_rows": len(train_samples),
            "validation_rows": len(validation_samples),
            "test_rows": len(test_samples),
            "test_size": config.test_size,
            "random_state": config.random_state,
            "split_strategy": "stratified_by_cefr_level",
        },
        "baseline_mean": baseline_metrics,
        "ridge_tfidf_readability_features": model_metrics,
        "calibrated_thresholds": final_model.cefr_thresholds,
        "selected_model": "ridge_tfidf_readability_features",
    }
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(config.model_path)
    config.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    """Точка входа для запуска обучения из командной строки."""

    parser = argparse.ArgumentParser(description="Обучение модели оценки сложности текста.")
    parser.add_argument("--config", default="configs/config.yaml", help="Путь к конфигурации проекта.")
    args = parser.parse_args()
    metrics = train(args.config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
