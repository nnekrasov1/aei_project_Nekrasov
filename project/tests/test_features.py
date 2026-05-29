import unittest

from src.text_complexity.features import (
    count_syllables,
    readability_complexity_score,
    readability_features,
    tokenize,
    vectorize,
)


class FeatureExtractionTest(unittest.TestCase):
    """Проверки извлечения признаков и вспомогательных текстовых функций."""

    def test_tokenize_keeps_simple_words(self) -> None:
        self.assertEqual(tokenize("The cat can't fly."), ["the", "cat", "can't", "fly"])

    def test_readability_features_have_expected_shape(self) -> None:
        # Проверяем не конкретные числа, а состав и базовую согласованность признаков.
        features = readability_features(
            "Simple sentences help new readers. Technical terminology makes passages harder."
        )

        self.assertGreater(features["word_count"], 5)
        self.assertEqual(features["sentence_count"], 2)
        self.assertGreater(features["avg_word_length"], 4)
        self.assertIn("flesch_reading_ease", features)
        self.assertIn("academic_word_ratio", features)
        self.assertIn("advanced_document_word_ratio", features)
        self.assertIn("subordinator_ratio", features)
        self.assertIn("basic_cefr_word_ratio", features)
        self.assertIn("out_of_basic_cefr_ratio", features)
        self.assertIn("formality_index", features)

    def test_vectorize_excludes_absolute_length_features(self) -> None:
        # Абсолютные длины не должны попадать в модельную часть признаков.
        features = vectorize("Tom lives in a small town. He buys apples.", ["tom", "small town"])

        self.assertNotIn("word_count", features)
        self.assertNotIn("sentence_count", features)
        self.assertIn("avg_sentence_length", features)
        self.assertIn("tfidf::tom", features)

    def test_advanced_document_words_are_counted(self) -> None:
        features = readability_features("Philanthropy can perpetuate structural inequality.")

        self.assertGreater(features["advanced_document_word_ratio"], 0.0)

    def test_syllable_estimator_is_reasonable(self) -> None:
        self.assertEqual(count_syllables("cat"), 1)
        self.assertGreaterEqual(count_syllables("methodological"), 5)

    def test_readability_complexity_grows_with_text_difficulty(self) -> None:
        # Эвристический readability-score должен сохранять естественный порядок сложности.
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

        self.assertLess(readability_complexity_score(easy), readability_complexity_score(medium))
        self.assertLess(readability_complexity_score(medium), readability_complexity_score(hard))


if __name__ == "__main__":
    unittest.main()
