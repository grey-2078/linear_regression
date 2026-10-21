# Предсказание погоды по открытым данным с сайта Open-Meteo


## Структура проекта

- `weather_loader.py` — загрузка данных Open-Meteo за выбранные годы.
- `load_data.py` — скрипт с вводом диапазона лет и координат.
- `model.py` — класс `GDRegressor`, реализующий градиентный спуск.
- `pipeline.py` — обучение, расчет метрик, поиск самых неточных прогнозов.
- `app.py` — приложение на Flask.
- `templates/index.html` — шаблон страницы.
- `static/style.css` — стили.
- `data/weather_dataset.csv` – сюда подгружаются данные.

## Источник данных

Open-Meteo: [https://open-meteo.com](https://open-meteo.com)
