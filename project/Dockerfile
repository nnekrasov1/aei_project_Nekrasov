FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=artifacts/text_complexity_model.json

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m src.text_complexity.train --config configs/config.yaml

EXPOSE 8000

CMD ["uvicorn", "src.text_complexity.service:app", "--host", "0.0.0.0", "--port", "8000"]
