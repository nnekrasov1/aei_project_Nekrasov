import unittest
from unittest.mock import patch

from src.text_complexity import service

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - тестовый модуль просто пропускается без FastAPI.
    TestClient = None


class FakeModel:
    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def predict(self, text: str) -> dict[str, object]:
        self.seen_texts.append(text)
        return {
            "score": 0.42,
            "level": "medium",
            "cefr_level": "B1",
            "confidence": 0.66,
            "features": {"word_count": 10.0},
        }


@unittest.skipIf(TestClient is None or service.app is None, "FastAPI не установлен")
class ServiceRuntimeTest(unittest.TestCase):
    """Проверки runtime-поведения FastAPI: UI и нормализация входного текста."""

    def test_home_page_is_available(self) -> None:
        fake_model = FakeModel()
        with patch.object(service, "load_runtime_model", return_value=fake_model):
            with TestClient(service.app) as client:
                response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Оценка сложности английского текста", response.text)
        self.assertIn("textarea", response.text)

    def test_predict_normalizes_multiline_text(self) -> None:
        fake_model = FakeModel()
        with patch.object(service, "load_runtime_model", return_value=fake_model):
            with TestClient(service.app) as client:
                response = client.post(
                    "/predict",
                    json={
                        "text": (
                            "Tom was very excited and ran back to the town.\n\n"
                            "The people in the town became very interested in the treasure."
                        )
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            fake_model.seen_texts[-1],
            "Tom was very excited and ran back to the town. "
            "The people in the town became very interested in the treasure.",
        )

    def test_predict_rejects_whitespace_only_text_after_normalization(self) -> None:
        fake_model = FakeModel()
        with patch.object(service, "load_runtime_model", return_value=fake_model):
            with TestClient(service.app) as client:
                response = client.post("/predict", json={"text": "\n \t \n"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Текст после нормализации оказался пустым")


if __name__ == "__main__":
    unittest.main()
