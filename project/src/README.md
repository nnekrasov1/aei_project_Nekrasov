# Исходный код

Основной код проекта находится в пакете `src/text_complexity`.

Модули:

- `config.py` - чтение простого YAML-конфига и разрешение путей;
- `data.py` - загрузка CSV, CEFR-маппинг и воспроизводимое train/test-разбиение;
- `features.py` - токенизация, readability-признаки, CEFR-лексические признаки и n-граммы;
- `model.py` - ridge-регрессия, CEFR-метрики, калибровка порогов, сохранение и загрузка модели;
- `prepare_cefr_dataset.py` - подготовка long-text CEFR-датасета в локальный CSV;
- `train.py` - CLI для стратифицированного обучения, калибровки порогов и сохранения артефактов;
- `service.py` - FastAPI-сервис с веб-страницей `/`, `/health`, `/predict`, нормализацией многострочного ввода и ограничением `10000` символов на запрос.

Запуск обучения из папки `project`:

```bash
python -m src.text_complexity.train --config configs/config.yaml
```

