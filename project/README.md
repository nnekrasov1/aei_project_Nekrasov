# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

В этой папке находится итоговый мини-проект по теме «Оценка уровня сложности текста».  
Проект демонстрирует применение методов и инструментов инженерии ИИ: работу с данными, выделение признаков, обучение модели, сравнение с baseline, API-сервис, Docker-сценарий и воспроизводимые эксперименты.

---

## 1. Паспорт проекта

- **Название проекта:** Оценка уровня сложности английского текста
- **Автор:** Некрасов Никита Алексеевич
- **Группа:** ИКБО-42-24
- **Контакт:** @Nekrsov

- **Краткое описание (2-4 предложения):**  
  Проект посвящён оценке сложности английского текста в терминах CEFR с акцентом на длинные связные материалы. Сервис принимает текст длиной до `10000` символов, автоматически нормализует переносы строк и лишние пробелы, вычисляет readability-признаки, CEFR-ориентированные лексические признаки и текстовые n-граммы, затем возвращает числовую оценку сложности от `0.0` до `1.0`, уровень `A1`-`C2`, категорию `easy` / `medium` / `hard`, уверенность и интерпретируемые признаки. В качестве финальной модели используется ridge-регрессия, обученная на long-text корпусе UniversalCEFR и дополнительно откалиброванная по CEFR-порогам на validation-части.

---

## 2. Структура проекта

Проект организован в следующей структуре:

- `requirements.txt` - зависимости проекта.
- `report.md` - отчёт по проекту: постановка задачи, данные, эксперименты, результаты.
- `self-checklist.md` - чеклист самопроверки проекта перед сдачей.
- `notebooks/` - экспериментальные ноутбуки:
  - `01_eda_and_experiments.ipynb` - EDA и описание экспериментального протокола.
- `src/` - основной код проекта:
  - `src/text_complexity/data.py` - загрузка данных и train/test-разбиение;
  - `src/text_complexity/features.py` - токенизация, readability-признаки, CEFR-лексика, n-граммы;
  - `src/text_complexity/model.py` - модель, метрики, сохранение и загрузка;
  - `src/text_complexity/train.py` - запуск обучения;
  - `src/text_complexity/service.py` - FastAPI-сервис.
- `data/` - данные проекта:
  - `cefr_long_en.csv` - основная выборка длинных `document-level` текстов из `UniversalCEFR/cambridge_exams_en` и `UniversalCEFR/elg_cefr_en`;
  - `sample_text_complexity.csv` - компактный fallback-датасет для быстрых unit-тестов;
  - персональные и конфиденциальные данные не используются.
- `configs/` - конфигурационные файлы:
  - `config.yaml` - пути, параметры разбиения и гиперпараметры;
  - `.env.example` - пример переменных окружения без секретов.
- `tests/` - модульные и sanity-тесты.
- `artifacts/` - сохранённая модель и метрики:
  - `text_complexity_model.json`;
  - `metrics.json`.
- `Dockerfile` и `.dockerignore` - Docker-сценарий запуска API.

Структура соответствует базовой структуре проекта из требований курса.

---

## 3. Требования и установка

### 3.1. Требования

- Python `>= 3.10`, рекомендуется Python `3.12`.
- Для локального API: зависимости из `requirements.txt`.
- Для контейнерного запуска: установленный Docker.

### 3.2. Установка окружения

```bash
# Перейти в папку проекта
cd project

# Создать виртуальное окружение
python -m venv .venv

# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

Дополнительные внешние сервисы не требуются. Модель обучается локально и сохраняется в `artifacts/`.

---

## 4. Как запустить проект

### 4.1. Запуск обучения модели

```bash
cd project
python -m src.text_complexity.train --config configs/config.yaml
```

Команда:

- читает `data/cefr_long_en.csv`;
- стратифицированно делит данные на train/validation/test;
- считает baseline;
- обучает ridge-регрессию на readability-признаках и TF-IDF-подобных n-граммах;
- калибрует CEFR-пороги на validation-части;
- сохраняет модель в `artifacts/text_complexity_model.json`;
- сохраняет метрики в `artifacts/metrics.json`.

Текущие метрики:

| Модель | MAE | RMSE | R2 | Spearman | Adjacent accuracy |
|---|---:|---:|---:|---:|---:|
| Mean baseline | 0.2256 | 0.2662 | - | - | - |
| Ridge TF-IDF + readability + CEFR features | 0.1182 | 0.1529 | 0.6701 | 0.8391 | 0.9103 |

### 4.2. Запуск сервиса (API)

Локальный запуск:

```bash
cd project
uvicorn src.text_complexity.service:app --host 0.0.0.0 --port 8000
```

Запуск через Docker:

```bash
cd project
docker build -t text-complexity-api .
docker run --rm -p 8000:8000 text-complexity-api
```

Сервис поднимается на порту `8000`.

Endpoints:

- `GET /` - простая веб-страница для вставки текста и просмотра результата модели.
- `GET /health` - health-check сервиса.
- `POST /predict` - оценка сложности английского текста в JSON-формате.
- `GET /docs` - Swagger UI.

Для перехода на веб-страницу перейдите: 
- http://localhost:8000/

Ограничения ввода:

- `POST /predict` принимает поле `text` длиной от `1` до `10000` символов.
- Переносы строк, абзацы и повторяющиеся пробелы автоматически нормализуются перед инференсом.

Пример проверки через `Invoke-RestMethod` для Windows PowerShell:

```bash
$body = @{
  text = "A great idea! Frazer and Peter are two 14-year-old boys who grew up in the same small Canadian town. They have always been friends and classmates. Like all their other friends, they enjoy going fishing or swimming at weekends. But for the last few months, they've spent every weekend in Peter's room working on his laptop. This isn't because they have a lot of homework. They have made a new computer word game. The idea for the game came from Frazer's little brother, Kevin, who had problems with his reading. Kevin learns words more easily by seeing pictures and hearing information than he does by reading. His brother wanted to help. Frazer and Peter worked together for over 200 hours to make a computer game and now it's ready to use. It's a speaking and picture game. For example, if you look at the word 'hat', there's a drawing of a hat next to it and you can hear Peter saying 'Hat! Hat!' at the same time. The two boys have won a lot of prizes for their computer game and it will soon be on sale around the world. Many schools are interested in buying it."
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://localhost:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Альтернатива через `curl.exe` в PowerShell:

```bash
curl.exe --request POST "http://localhost:8000/predict" `
  --header "Content-Type: application/json" `
  --data-raw "{""text"":""A great idea! Frazer and Peter are two 14-year-old boys who grew up in the same small Canadian town. They have always been friends and classmates. Like all their other friends, they enjoy going fishing or swimming at weekends. But for the last few months, they've spent every weekend in Peter's room working on his laptop. This isn't because they have a lot of homework. They have made a new computer word game. The idea for the game came from Frazer's little brother, Kevin, who had problems with his reading. Kevin learns words more easily by seeing pictures and hearing information than he does by reading. His brother wanted to help. Frazer and Peter worked together for over 200 hours to make a computer game and now it's ready to use. It's a speaking and picture game. For example, if you look at the word 'hat', there's a drawing of a hat next to it and you can hear Peter saying 'Hat! Hat!' at the same time. The two boys have won a lot of prizes for their computer game and it will soon be on sale around the world. Many schools are interested in buying it.""}"
```

Пример ответа:

```json
{
  "score": 0.307,
  "level": "medium",
  "cefr_level": "B1",
  "confidence": 0.5559,
  "features": {
    "word_count": 197,
    "sentence_count": 17,
    "avg_sentence_length": 11.588235294117647,
    "avg_word_length": 4.243654822335025,
    "syllables_per_word": 1.3299492385786802,
    "lexical_diversity": 0.6294416243654822,
    "long_word_ratio": 0.16751269035532995,
    "very_long_word_ratio": 0.015228426395939087,
    "subordinator_ratio": 0.01015228426395939,
    "connector_ratio": 0,
    "modal_ratio": 0.01015228426395939,
    "academic_word_ratio": 0,
    "advanced_document_word_ratio": 0,
    "basic_cefr_word_ratio": 0.47715736040609136,
    "intermediate_cefr_word_ratio": 0.005076142131979695,
    "out_of_basic_cefr_ratio": 0.5228426395939086,
    "nominalization_ratio": 0.005076142131979695,
    "punctuation_density": 0.030456852791878174,
    "formality_index": 0.07969543147208123,
    "flesch_reading_ease": 82.55923559271427
  }
}
```

---

## 5. Данные

Используется только английский текст.

- Источники: `UniversalCEFR/cambridge_exams_en`, `UniversalCEFR/elg_cefr_en`.
- Файл для обучения: `data/cefr_long_en.csv`.
- Назначение выборки: long-text корпус для оценки длинных связных материалов без коротких примеров-предложений.
- Текущий размер подготовленной выборки: `779` текстов, все записи имеют формат `document-level`.
- Лицензия источника: `CC BY-NC-SA 4.0`, поэтому проект следует рассматривать как учебную и некоммерческую демонстрацию.
- Большие файлы, персональные данные и конфиденциальные данные в репозитории не хранятся.

Структура данных:

| Колонка | Описание |
|---|---|
| `text` | английский текст |
| `cefr_level` | исходный уровень CEFR: `A1`, `A2`, `B1`, `B2`, `C1`, `C2` |
| `target` | числовая сложность от `0.0` до `1.0` |
| `source_name` | название исходного корпуса |
| `format` | гранулярность текста; в текущем датасете используется `document-level` |
| `category` | тип текста: `reference` или `learner` |
| `license` | лицензия исходного корпуса |

CSV формируется командой:

```bash
python -m src.text_complexity.prepare_cefr_dataset
```

По умолчанию скрипт собирает long-text датасет. Опции `--long-max-per-level` и `--min-words` управляют балансом по уровням и минимальной длиной текстов.

---

## 6. Тесты

Тесты проверяют:

- токенизацию, подсчёт слогов и readability-признаки;
- обучение модели;
- сохранение и загрузку модели;
- правила перевода score в категории сложности.

Команда запуска:

```bash
cd project
python -m unittest discover tests
```

---

## 7. Демонстрация на защите

На защите планируется:

1. Кратко показать структуру проекта: `data/`, `notebooks/`, `src/`, `configs/`, `artifacts/`, `tests/`.
2. Запустить обучение:

   ```bash
   python -m src.text_complexity.train --config configs/config.yaml
   ```

3. Показать метрики baseline и финальной модели из `artifacts/metrics.json`.
4. Запустить API локально или через Docker.
5. Открыть `http://localhost:8000/` и показать ввод текста через встроенную веб-страницу.
6. Отправить 2-3 запроса в `/predict` или через страницу `/` с простым, средним и сложным английским текстом.
7. Показать, что `/predict` использует сохранённую модель, а не заглушку.

---

## 8. Ограничения и дальнейшая работа

Текущие ограничения:

- датасет подготовлен с ограничением числа строк на уровень, поэтому это учебная версия, а не максимально возможная модель;
- модель в первую очередь настроена под длинные связные тексты, поэтому на очень коротких предложениях возможна заметная деградация;
- датасет относительно небольшой, поэтому метрики нельзя считать заменой внешней промышленной валидации;
- сервис не хранит историю запросов и не имеет авторизации;
- модель использует классические признаки, без embeddings и transformer-моделей.
- модель имеет лимит 10000 символов.

Дальнейшая работа:

- расширить document-level часть UniversalCEFR или подключить другие англоязычные CEFR-корпуса;
- добавить кросс-валидацию;
- сравнить ridge-регрессию с gradient boosting и transformer embeddings;
- добавить e2e-тесты API;
- расширить наблюдаемость и хранение истории запросов.

---

## 9. Оценка проекта

Итоговая оценка за проект выставляется по пятибалльной шкале (2-5).

Ориентиры для оценки:

- **2** - проект не принят:
  - не выполняются минимальные требования;
  - сервис не запускается или отсутствует ключевой функционал;
  - есть грубые нарушения требований курса.
- **3** - проект принят, но реализован на базовом уровне:
  - минимальный функционал есть;
  - по чеклисту выполнено меньше 5 пунктов.
- **4** - хороший, рабочий проект:
  - сервис запускается по `README.md`;
  - `/predict` использует реальную модель;
  - есть данные, EDA и эксперименты с метриками;
  - по чеклисту выполнено не менее 5 пунктов.
- **5** - сильный, хорошо проработанный проект:
  - аккуратно реализован сервис и пайплайн;
  - проведены осмысленные эксперименты;
  - обоснован выбор финальной модели;
  - есть конфиги, `.env.example`, Docker и `/health`;
  - документация позволяет быстро воспроизвести решение;
  - по чеклисту выполнено не менее 9 пунктов.

Чеклист `self-checklist.md` служит для самопроверки студента и как подсказка при проверке.  
Окончательное решение по оценке остаётся за преподавателем и может учитывать:

- качество реализации внутри каждого пункта чеклиста;
- дополнительные сильные стороны проекта (нестандартные решения, дополнительные функции, продвинутая ML-часть и т.п.);
- соблюдение требований курса и дедлайнов.

---
