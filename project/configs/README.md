# Конфигурация

`config.yaml` управляет обучением и путями артефактов:

- `data_path` - CSV с колонками `text`, `cefr_level` и `target`;
- `model_path` - куда сохранить обученную модель;
- `metrics_path` - куда сохранить метрики эксперимента;
- `test_size` - доля тестовой выборки;
- `random_state` - seed для воспроизводимого разбиения;
- `max_features` - максимум текстовых n-грамм;
- `regularization` - L2-регуляризация ridge-регрессии;
- `epochs` - число эпох оптимизации.

Текущий основной датасет: `data/cefr_long_en.csv`, подготовленный из `UniversalCEFR/cambridge_exams_en` и `UniversalCEFR/elg_cefr_en` под long-text сценарий.

`.env.example` показывает переменные окружения для запуска сервиса. Реальные `.env`-файлы и секреты не должны попадать в репозиторий.

