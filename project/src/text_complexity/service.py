from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - позволяет гонять тесты без обязательного FastAPI.
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]
    HTMLResponse = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        return None

from .model import TextComplexityModel

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
DEMO_PAGE_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Оценка сложности текста</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f1ea;
      --panel: #fffdf8;
      --ink: #1f1b16;
      --muted: #6c6257;
      --accent: #186b5b;
      --accent-strong: #0f4f43;
      --line: #d9cec2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(24, 107, 91, 0.16), transparent 32%),
        linear-gradient(180deg, #fbf8f2 0%, var(--bg) 100%);
      color: var(--ink);
    }
    main {
      max-width: 1040px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    .hero {
      display: grid;
      gap: 12px;
      margin-bottom: 24px;
    }
    h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.1rem);
      line-height: 1.05;
    }
    .subtitle {
      max-width: 760px;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.5;
    }
    .layout {
      display: grid;
      gap: 20px;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      align-items: start;
    }
    .panel {
      background: rgba(255, 253, 248, 0.94);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 40px rgba(31, 27, 22, 0.08);
      padding: 18px;
    }
    textarea {
      width: 100%;
      min-height: 340px;
      resize: vertical;
      border: 1px solid #c9beb1;
      border-radius: 8px;
      padding: 14px;
      font: inherit;
      line-height: 1.6;
      color: var(--ink);
      background: #fffdfa;
    }
    textarea:focus {
      outline: 2px solid rgba(24, 107, 91, 0.18);
      border-color: var(--accent);
    }
    .actions {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 14px;
    }
    button {
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      padding: 12px 18px;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled {
      background: #87a89f;
      cursor: wait;
    }
    .status {
      color: var(--muted);
      min-height: 1.5rem;
    }
    .summary {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-bottom: 16px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fffaf2;
    }
    .metric-label {
      color: var(--muted);
      font-size: 0.82rem;
      margin-bottom: 6px;
    }
    .metric-value {
      font-size: 1.28rem;
      font-weight: 700;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }
    th, td {
      text-align: left;
      padding: 8px 0;
      border-bottom: 1px solid #eee4d7;
      vertical-align: top;
    }
    th {
      font-weight: 600;
      color: var(--muted);
    }
    .empty {
      color: var(--muted);
      line-height: 1.5;
    }
    @media (max-width: 880px) {
      .layout { grid-template-columns: 1fr; }
      .summary { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 560px) {
      .summary { grid-template-columns: 1fr; }
      main { padding: 20px 14px 36px; }
      textarea { min-height: 260px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Оценка сложности английского текста</h1>
      <div class="subtitle">
        Вставьте текст. Сервис отправит запрос в модель и 
        покажет итоговый уровень вместе с интерпретируемыми признаками.
      </div>
    </section>
    <section class="layout">
      <div class="panel">
        <textarea id="text-input" placeholder="Введите английский текст для анализа..."></textarea>
        <div class="actions">
          <button id="predict-button" type="button">Оценить</button>
          <div class="status" id="status"></div>
        </div>
      </div>
      <div class="panel">
        <div class="summary" id="summary">
          <div class="metric">
            <div class="metric-label">Статус</div>
            <div class="metric-value">Ожидание</div>
          </div>
        </div>
        <div id="details" class="empty">После запроса здесь появятся score, уровни и признаки модели.</div>
      </div>
    </section>
  </main>
  <script>
    const textInput = document.getElementById("text-input");
    const predictButton = document.getElementById("predict-button");
    const statusNode = document.getElementById("status");
    const summaryNode = document.getElementById("summary");
    const detailsNode = document.getElementById("details");

    function renderSummary(prediction) {
      summaryNode.innerHTML = `
        <div class="metric"><div class="metric-label">Score</div><div class="metric-value">${prediction.score}</div></div>
        <div class="metric"><div class="metric-label">CEFR</div><div class="metric-value">${prediction.cefr_level}</div></div>
        <div class="metric"><div class="metric-label">Категория</div><div class="metric-value">${prediction.level}</div></div>
        <div class="metric"><div class="metric-label">Уверенность</div><div class="metric-value">${prediction.confidence}</div></div>
      `;
    }

    function renderFeatures(features) {
      const rows = Object.entries(features)
        .map(([name, value]) => `<tr><td>${name}</td><td>${value}</td></tr>`)
        .join("");
      detailsNode.className = "";
      detailsNode.innerHTML = `<table><thead><tr><th>Признак</th><th>Значение</th></tr></thead><tbody>${rows}</tbody></table>`;
    }

    async function runPrediction() {
      const text = textInput.value;
      if (!text.trim()) {
        statusNode.textContent = "Нужен непустой текст.";
        return;
      }
      predictButton.disabled = true;
      statusNode.textContent = "Считаю...";
      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Не удалось получить предсказание.");
        }
        renderSummary(payload);
        renderFeatures(payload.features);
        statusNode.textContent = "Готово.";
      } catch (error) {
        statusNode.textContent = error.message;
      } finally {
        predictButton.disabled = false;
      }
    }

    predictButton.addEventListener("click", runPrediction);
  </script>
</body>
</html>
"""


class PredictionRequest(BaseModel):
    """Тело запроса для /predict."""

    text: str = Field(..., min_length=1, max_length=10000)


class PredictionResponse(BaseModel):
    """Схема ответа API с итоговым score и интерпретируемыми признаками."""

    score: float
    level: str
    cefr_level: str
    confidence: float
    features: dict[str, float]


def normalize_input_text(text: str) -> str:
    """Приводит текст к однострочному виду, сохраняя слова и границы предложений."""

    return " ".join(text.split())


def load_runtime_model() -> TextComplexityModel:
    """Загружает модель из пути, заданного в переменной окружения."""

    model_path = Path(os.getenv("MODEL_PATH", "artifacts/text_complexity_model.json"))
    if not model_path.exists():
        raise FileNotFoundError(f"Артефакт модели не найден: {model_path}")
    return TextComplexityModel.load(model_path)


if FastAPI is not None:
    app = FastAPI(title="API оценки сложности текста", version="0.1.0")
    _model: TextComplexityModel | None = None

    @app.on_event("startup")
    def startup() -> None:
        global _model
        # Модель загружается один раз при старте приложения,
        # чтобы не читать артефакт заново на каждый запрос.
        _model = load_runtime_model()
        logger.info("Модель оценки сложности текста загружена")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return DEMO_PAGE_HTML

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> dict[str, object]:
        if _model is None:
            raise HTTPException(status_code=503, detail="Модель еще не загружена")
        normalized_text = normalize_input_text(request.text)
        if not normalized_text:
            raise HTTPException(status_code=422, detail="Текст после нормализации оказался пустым")
        # Вся бизнес-логика предсказания живёт в модели;
        # сервис только валидирует вход и отдаёт сериализованный ответ.
        prediction = _model.predict(normalized_text)
        logger.info("Предсказание выполнено: level=%s score=%s", prediction["level"], prediction["score"])
        return prediction
else:
    app = None
