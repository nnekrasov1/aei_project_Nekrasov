import unittest

from src.text_complexity.model import confidence_from_score, score_to_cefr_level, score_to_level


class ServiceContractTest(unittest.TestCase):
    """Проверки контракта API-логики: уровни, границы и уверенность."""

    def test_score_to_level_contract(self) -> None:
        self.assertEqual(score_to_level(0.10), "easy")
        self.assertEqual(score_to_level(0.50), "medium")
        self.assertEqual(score_to_level(0.90), "hard")

    def test_score_to_cefr_level_contract(self) -> None:
        self.assertEqual(score_to_cefr_level(0.00), "A1")
        self.assertEqual(score_to_cefr_level(0.25), "A2")
        self.assertEqual(score_to_cefr_level(0.45), "B1")
        self.assertEqual(score_to_cefr_level(0.65), "B2")
        self.assertEqual(score_to_cefr_level(0.85), "C1")
        self.assertEqual(score_to_cefr_level(1.00), "C2")

    def test_score_to_cefr_level_rounds_borderline_scores(self) -> None:
        self.assertEqual(score_to_cefr_level(0.094), "A1")
        self.assertEqual(score_to_cefr_level(0.095), "A2")
        self.assertEqual(score_to_cefr_level(0.699), "C1")
        self.assertEqual(score_to_cefr_level(0.899), "C2")

    def test_score_to_cefr_level_supports_calibrated_thresholds(self) -> None:
        # После калибровки пороги могут отличаться от стандартной таблицы.
        thresholds = [0.16, 0.32, 0.48, 0.64, 0.84]

        self.assertEqual(score_to_cefr_level(0.15, thresholds), "A1")
        self.assertEqual(score_to_cefr_level(0.16, thresholds), "A2")
        self.assertEqual(score_to_cefr_level(0.65, thresholds), "C1")

    def test_confidence_tracks_distance_from_custom_boundaries(self) -> None:
        # Чем дальше score от границы уровня, тем выше должна быть уверенность.
        thresholds = [0.16, 0.32, 0.48, 0.64, 0.84]

        self.assertLess(confidence_from_score(0.64, thresholds), confidence_from_score(0.75, thresholds))


if __name__ == "__main__":
    unittest.main()
