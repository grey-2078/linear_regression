import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from pathlib import Path

from pipeline import FEATURE_COLUMNS, train_model
from weather_loader import fetch_weather_for_year_range, save_dataset

app = Flask(__name__)
MODEL = None
METRICS: dict[str, float] = {}
WEIGHTS: dict[str, float] = {}
WORST_RECORDS: list[dict] = []
LAST_PREDICTION: float | None = None
DATASET_PATH = Path("data/weather_dataset.csv")
TRAINING_CONFIG = {
    "start_year": 2022,
    "end_year": 2026,
    "latitude": 55.75,
    "longitude": 30.19,
}
SCALER = None


def training(
        start_year: int = 2022,
        end_year: int = 2024,
        latitude: float = 59.57,
        longitude: float = 30.19,
) -> None:
    global MODEL, METRICS, WEIGHTS, WORST_RECORDS, LAST_PREDICTION, TRAINING_CONFIG, SCALER

    data = fetch_weather_for_year_range(
        latitude,
        longitude,
        start_year=start_year,
        end_year=end_year,
    )
    save_dataset(data, DATASET_PATH)

    #запуск пайплайна обучения модели
    reg, metrics, anomalies, scaler = train_model(data)
    MODEL = reg
    METRICS = metrics
    SCALER = scaler

    # собираем словарь весов: {имя_признака: значение_коэффициента} + Свободный член
    WEIGHTS = {FEATURE_COLUMNS[i]: float(reg.coef_[i, 0]) for i in range(len(FEATURE_COLUMNS))}
    WEIGHTS["intercept"] = float(reg.intercept_)

    WORST_RECORDS = anomalies.to_dict(orient="records")
    LAST_PREDICTION = None
    TRAINING_CONFIG = {
        "start_year": start_year,
        "end_year": end_year,
        "latitude": latitude,
        "longitude": longitude,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    global LAST_PREDICTION

    if request.method == "POST":
        action = request.form.get("action")
        if action == "retrain":
            start_year = int(request.form["start_year"])
            end_year = int(request.form["end_year"])
            latitude = float(request.form["latitude"])
            longitude = float(request.form["longitude"])
            training(
                start_year=start_year,
                end_year=end_year,
                latitude=latitude,
                longitude=longitude,
            )
        elif action == "predict":
            user_input = np.array(
                [
                    [
                        float(request.form["temperature_2m"]),
                        float(request.form["relative_humidity_2m"]),
                        float(request.form["wind_speed_10m"]),
                    ]
                ]
            )
            # делаем предсказание (встроенный scaler внутри MODEL отработает сам)
            # берем [0], так как predict возвращает массив предсказаний
            LAST_PREDICTION = float(MODEL.predict(user_input)[0])

    return render_template(
        "index.html",
        metrics=METRICS,
        weights=WEIGHTS,
        prediction=LAST_PREDICTION,
        worst_records=WORST_RECORDS,
        training_config=TRAINING_CONFIG,
    )


if __name__ == "__main__":
    if DATASET_PATH.exists():
        data = pd.read_csv(DATASET_PATH, parse_dates=["date"])
        # запуск пайплайна на уже существующем датасете
        reg, metrics, anomalies, scaler = train_model(data)
        MODEL = reg
        METRICS = metrics
        SCALER = scaler

        # заполняем веса для начальной загрузки из файла
        WEIGHTS = {FEATURE_COLUMNS[i]: float(reg.coef_[i, 0]) for i in range(len(FEATURE_COLUMNS))}
        WEIGHTS["intercept"] = float(reg.intercept_)

        WORST_RECORDS = anomalies.to_dict(orient="records")
    else:
        training(
            start_year=TRAINING_CONFIG["start_year"],
            end_year=TRAINING_CONFIG["end_year"],
            latitude=TRAINING_CONFIG["latitude"],
            longitude=TRAINING_CONFIG["longitude"]
        )
    app.run(debug=True)
