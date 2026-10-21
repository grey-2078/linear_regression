import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from model import GDRegressor

FEATURE_COLUMNS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
TARGET_COLUMN = "apparent_temperature"


def z_scaler(feature):
    """Стандартизация Z-score для многомерных массивов."""
    X = np.array(feature, dtype=np.float64)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0
    return (X - np.mean(X, axis=0)) / stds


def rmse(y_true, y_pred):
    """Корень из среднеквадратичной ошибки."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def r_squared(y_true, y_pred):
    """Коэффициент детерминации R^2."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_bar = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_bar) ** 2)
    return 1 - (ss_res / ss_tot)


def find_optimal_params(X, y):
    """Подбирает лучшие параметры n_iter и alpha для масштабированных данных."""
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=18)

    best_rmse = float('inf')
    best_alpha = 0
    best_n_iter = 0

    alphas = [0.01, 0.05, 0.1, 0.2]
    n_iters = [100, 300, 500]

    for a in alphas:
        for n in n_iters:
            tmp_model = GDRegressor(alpha=a, n_iter=n, progress=False)
            tmp_model.fit(X_train, Y_train)
            predictions = tmp_model.predict(X_test)

            current_rmse = rmse(Y_test, predictions)

            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_alpha = a
                best_n_iter = n

    return best_n_iter, best_alpha


def train_model(data: pd.DataFrame):
    """В этой функции описывается пайплайн обучения вашей модели линейной регрессии.
    - reg – обученная модель
    - anomalies –самые неточные предсказания в тестовой выборке.
    - scaler – функция стандартизации данных."""
    if data.empty:
        raise ValueError("Данных нет")

    # выделяем признаки и целевую переменную
    X = data[FEATURE_COLUMNS].values
    y = data[TARGET_COLUMN].values.reshape(-1, 1)

    # разделяем выборку на обучающую и тестовую
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=18
    )

    scaler = z_scaler

    # подбираем параметры и обучаем модель
    max_iter, alpha = find_optimal_params(X, y)
    reg = GDRegressor(alpha=alpha, n_iter=max_iter, progress=False)
    reg.fit(X_train, y_train)

    # предсказания на тестовой выборке
    y_pred = reg.predict(X_test)
    y_test_flat = y_test.flatten()

    # считаем метрики качества
    mae_val = np.mean(np.abs(y_test_flat - y_pred))
    rmse_val = rmse(y_test_flat, y_pred)
    r2_val = r_squared(y_test_flat, y_pred)

    # поиск аномалий (самых неточных предсказаний)
    absolute_errors = np.abs(y_test_flat - y_pred)

    # аномалиями считаем ошибки, превышающие 3 * RMSE
    anomaly_threshold = 3 * rmse_val
    anomaly_mask = absolute_errors > anomaly_threshold

    # получаем индексы строк из исходного DataFrame, которые попали в аномалии
    # берем кортеж под индексом [1], так как нам нужен именно тест-сплит для индексов
    test_indices = data.iloc[train_test_split(data.index, test_size=0.33, random_state=18)[1]].index

    # фильтруем исходный DataFrame, оставляя только аномальные строки
    anomalies = data.loc[test_indices[anomaly_mask]]

    return reg, {"MAE": mae_val, "RMSE": rmse_val, "R2": r2_val}, anomalies, scaler
