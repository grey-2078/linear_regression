import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


class GDRegressor:
    """
    Скопируйте или наследуйте класс GDRegressor из файла linreg_housing/linreg.py,
    адаптировав обучение под многомерную регрессию с нефиксированным числом признаков.
    """

    def __init__(self, alpha=0.001, n_iter=100, progress=False):
        self.alpha = alpha
        self.n_iter = n_iter
        self.progress = progress
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []
        self.theta_history = []
        #сохраняем параметры масштабирования для произвольного n
        self.means_ = None
        self.stds_ = None

    def fit(self, X_train, y_train):
        #делаем двумерность матрицы признаков (m объектов, n признаков)
        X = np.array(X_train, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y = np.array(y_train, dtype=np.float64).reshape(-1, 1)
        m, n = X.shape

        #многомерное масштабирование по колонкам (axis=0)
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)
        self.stds_[self.stds_ == 0] = 1.0  #защита от деления на ноль
        X_scaled = (X - self.means_) / self.stds_

        #инициализация параметров под n признаков
        self.coef_ = np.zeros((n, 1))
        self.intercept_ = 0.0

        #динамически создаем структуру истории под текущее число признаков n
        self.loss_history = []
        self.theta_history = [[] for _ in range(n + 1)]

        iterator = range(self.n_iter)
        if self.progress:
            iterator = tqdm(iterator, desc="Training")

        for _ in iterator:
            #предсказание для n признаков X (m x n) * coef_ (n x 1) + b
            y_hat = np.dot(X_scaled, self.coef_) + self.intercept_
            error = y_hat - y

            #градиенты (размерность grad_w совпадает с coef_: n x 1)
            grad_w = (1 / m) * np.dot(X_scaled.T, error)
            grad_b = (1 / m) * np.sum(error)

            #обновление весов
            self.coef_ -= self.alpha * grad_w
            self.intercept_ -= self.alpha * grad_b

            #запись истории
            loss = (1 / (2 * m)) * np.sum(error ** 2)
            self.loss_history.append(loss)

            self.theta_history[0].append(self.intercept_)
            #индексы от 1 до n — коэффициенты для каждого из n признаков
            for i in range(n):
                self.theta_history[i + 1].append(self.coef_[i, 0])

    def predict(self, X_test):
        X = np.array(X_test, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        #масштабируем тест по параметрам из fit
        X_scaled = (X - self.means_) / self.stds_
        return (np.dot(X_scaled, self.coef_) + self.intercept_).flatten()
