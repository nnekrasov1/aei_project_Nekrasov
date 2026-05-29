# Данные

Проект работает только с английскими текстами и ориентирован на оценку уровня сложности в терминах CEFR.

Основной датасет:

- `cefr_long_en.csv` - выборка длинных document-level английских текстов UniversalCEFR.
- `raw/*.json` - исходные JSON-файлы, скачанные с Hugging Face.
- `sample_text_complexity.csv` - маленький fallback-датасет для быстрых тестов и демонстрации формата.

Формат `cefr_long_en.csv`:

| Колонка | Тип | Описание |
|---|---|---|
| `text` | string | английский связный текст |
| `cefr_level` | string | уровень `A1`, `A2`, `B1`, `B2`, `C1` или `C2` |
| `target` | float | числовая сложность от `0.0` до `1.0` |
| `source_name` | string | название исходного корпуса |
| `format` | string | гранулярность текста; в текущем датасете используется `document-level` |
| `category` | string | тип текста: `reference` или `learner` |
| `license` | string | лицензия исходного корпуса |

Подготовка данных:

```bash
python -m src.text_complexity.prepare_cefr_dataset
```

По умолчанию скрипт собирает long-text датасет из `cambridge_exams_en` и `elg_cefr_en`, затем сохраняет результат в `data/cefr_long_en.csv`.

Текущая подготовленная версия содержит 779 строк long-text формата.

Распределение по уровням:

- `A1` - 25
- `A2` - 144
- `B1` - 221
- `B2` - 179
- `C1` - 123
- `C2` - 87

Источники: `UniversalCEFR/cambridge_exams_en`, `UniversalCEFR/elg_cefr_en`. Лицензии источников некоммерческие (`CC BY-NC-SA 4.0` / совместимые), поэтому проект следует использовать как учебную и некоммерческую демонстрацию.

