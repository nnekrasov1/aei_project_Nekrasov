import tempfile
import unittest
from pathlib import Path

from src.text_complexity.config import AppConfig
from src.text_complexity.data import cefr_to_target, load_dataset, train_test_split
from src.text_complexity.prepare_cefr_dataset import convert_cefr_json_to_csv
from src.text_complexity.model import (
    TextComplexityModel,
    calibrate_cefr_thresholds,
    evaluate,
    spearman_correlation,
    train_ridge_regression,
)


class TrainingPipelineTest(unittest.TestCase):
    """Проверки обучающего пайплайна и сериализации модели."""

    def test_model_trains_predicts_and_serializes(self) -> None:
        samples = load_dataset("data/sample_text_complexity.csv")
        train_samples, test_samples = train_test_split(samples, test_size=0.25, random_state=42)
        model = train_ridge_regression(
            texts=[sample.text for sample in train_samples],
            targets=[sample.target for sample in train_samples],
            max_features=60,
            regularization=1.0,
            epochs=100,
        )

        metrics = evaluate(
            model,
            texts=[sample.text for sample in test_samples],
            targets=[sample.target for sample in test_samples],
        )
        prediction = model.predict("The methodological framework complicates causal interpretation.")

        self.assertIn(prediction["level"], {"easy", "medium", "hard"})
        self.assertIn(prediction["cefr_level"], {"A1", "A2", "B1", "B2", "C1", "C2"})
        self.assertGreaterEqual(prediction["score"], 0.0)
        self.assertLessEqual(prediction["score"], 1.0)
        self.assertIn("mae", metrics)
        self.assertIn("spearman", metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # После сохранения и загрузки модель должна сохранить ту же структуру признаков.
            path = Path(tmp_dir) / "model.json"
            model.save(path)
            loaded = TextComplexityModel.load(path)
            self.assertEqual(loaded.vocabulary, model.vocabulary)

    def test_model_orders_a1_b1_c1_examples(self) -> None:
        samples = load_dataset("data/sample_text_complexity.csv")
        model = train_ridge_regression(
            texts=[sample.text for sample in samples],
            targets=[sample.target for sample in samples],
            max_features=100,
            regularization=1.0,
            epochs=200,
        )
        easy = (
            "My cat is very small and soft. She likes to sleep in the sun all day. "
            "Every afternoon, I give her some milk and a little fish. "
            "She is a very happy cat, and I love her very much."
        )
        medium = (
            "Although many people prefer to travel by plane because it is faster, "
            "I personally enjoy taking the train. It allows you to see the changing "
            "landscape while you relax. Besides, you do not have to worry about long "
            "security lines at the airport, which makes the journey much less stressful."
        )
        hard = (
            "The rapid proliferation of artificial intelligence has precipitated a profound "
            "shift in contemporary labor markets, challenging conventional paradigms of "
            "professional expertise. While proponents argue that automation fosters "
            "unprecedented efficiency, skeptics contend that the erosion of human-centric "
            "roles could exacerbate socioeconomic disparities, necessitating a comprehensive "
            "re-evaluation of our educational frameworks."
        )

        predictions = [model.predict(text) for text in [easy, medium, hard]]

        # На маленьком toy-наборе допустима погрешность по ярлыку,
        # но порядок score обязан оставаться осмысленным.
        self.assertIn(predictions[0]["level"], {"easy", "medium"})
        self.assertEqual(predictions[1]["level"], "medium")
        self.assertEqual(predictions[2]["level"], "hard")
        self.assertLess(predictions[0]["score"], predictions[1]["score"])
        self.assertLess(predictions[1]["score"], predictions[2]["score"])

    def test_config_dataclass_paths(self) -> None:
        config = AppConfig(
            data_path=Path("data/sample_text_complexity.csv"),
            model_path=Path("artifacts/model.json"),
            metrics_path=Path("artifacts/metrics.json"),
        )
        self.assertEqual(config.test_size, 0.25)

    def test_cefr_target_mapping(self) -> None:
        self.assertEqual(cefr_to_target("A1"), 0.0)
        self.assertEqual(cefr_to_target("A1+"), 0.0)
        self.assertEqual(cefr_to_target("C2"), 1.0)

    def test_prepare_dataset_can_filter_long_texts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "raw.json"
            output_path = Path(tmp_dir) / "prepared.csv"
            raw_path.write_text(
                """[
                  {"text": "Short text.", "cefr_level": "A2", "format": "document-level"},
                  {"text": "This document has enough words to pass the long text filter and should be included in the prepared output.", "cefr_level": "B1", "format": "document-level"},
                  {"text": "This sentence-level item has enough words but the wrong format for the long text filter.", "cefr_level": "B2", "format": "sentence-level"}
                ]""",
                encoding="utf-8",
            )

            count = convert_cefr_json_to_csv(
                raw_path,
                output_path,
                formats={"document-level"},
                min_words=10,
            )
            rows = load_dataset(output_path)

        # В итоговый CSV должен пройти только длинный документ нужного формата.
        self.assertEqual(count, 1)
        self.assertEqual(rows[0].level, "B1")

    def test_spearman_correlation_rewards_correct_order(self) -> None:
        self.assertAlmostEqual(
            spearman_correlation([0.0, 0.5, 1.0], [0.1, 0.6, 0.9]),
            1.0,
        )
        self.assertLess(spearman_correlation([0.0, 0.5, 1.0], [0.9, 0.6, 0.1]), 0.0)

    def test_threshold_calibration_returns_monotonic_boundaries(self) -> None:
        samples = load_dataset("data/sample_text_complexity.csv")
        model = train_ridge_regression(
            texts=[sample.text for sample in samples],
            targets=[sample.target for sample in samples],
            max_features=80,
            regularization=1.0,
            epochs=150,
        )

        thresholds = calibrate_cefr_thresholds(
            model,
            texts=[sample.text for sample in samples],
            levels=[sample.level for sample in samples],
        )

        self.assertEqual(len(thresholds), 5)
        self.assertTrue(all(left < right for left, right in zip(thresholds, thresholds[1:])))

    def test_train_test_split_preserves_all_levels_in_holdout(self) -> None:
        # Стратификация нужна, чтобы даже на маленькой выборке уровни не пропадали из holdout.
        duplicated = load_dataset("data/sample_text_complexity.csv") * 2
        _, test_samples = train_test_split(duplicated, test_size=0.4, random_state=42)

        self.assertGreaterEqual(len({sample.level for sample in test_samples}), 3)


if __name__ == "__main__":
    unittest.main()
