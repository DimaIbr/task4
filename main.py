import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

df = pd.read_csv('lin_reg.txt', header=None)
df.rename(
    columns={
        0: 'x1',
        1: 'x2',
        2: 'x3',
        3: 'x4',
        4: 'y',
    },
    inplace=True
)

X = df[['x1', 'x2', 'x3', 'x4']].values
Y = df[['y']].values


def forward_pass(w, X):
    b = w[0] * np.ones((X.shape[0], 1))
    w = w[1:]
    return (X @ w).reshape(-1, 1) + b.reshape(-1, 1)


# функция потерь логистической регрессии с регуляризатором с параметром а
def func(w, X, Y, a=0):
    return np.sum(np.power(forward_pass(w, X) - Y, 2)) / len(Y) + a * (np.sum(np.power(w, 2)))


# разделение на выбоки
def train_test_split(X, train_size=0.8):
    n = X.shape[0]
    pivot = int(n * train_size)
    return X[:pivot], X[pivot:]


def train_test_split_df(df):
    train, test = train_test_split(df)
    X_train = train[['x1', 'x2', 'x3', 'x4']].values
    X_test = test[['x1', 'x2', 'x3', 'x4']].values
    Y_train = train[['y']].values
    Y_test = test[['y']].values
    return X_train, X_test, Y_train, Y_test


# Обучение
def fit_linear(X, Y, a=0, plot=False):
    history = {'w': [], 'error': []}

    def callback(w):
        history['w'].append(w)
        history['error'].append(func(w, X, Y))

    optimal_w = minimize(func, x0=np.zeros((X.shape[1] + 1, 1)), args=(X, Y, a), callback=callback, method='BFGS').x

    return optimal_w, history


# Кросс-валидация

def cross_validation(df, cv=5, a=0):
    shuffled_df = df.sample(frac=1)
    n = len(shuffled_df)
    step = n // cv

    record = {
        'train_error': [],
        'test_error': []
    }
    for i in range(0, n, step):
        fold = shuffled_df[i:i + step]
        X_train, X_test, Y_train, Y_test = train_test_split_df(fold)
        w_optimal, _ = fit_linear(X_train, Y_train, a=a)
        record['train_error'].append(func(w_optimal, X_train, Y_train))
        record['test_error'].append(func(w_optimal, X_test, Y_test))
    return record


def mean_cv_record(record):
    return np.array(record['test_error']).mean()


# Кривая  обучения
def plot_learning_curve(record, X_test, Y_test, optimal_a):
    plt.figure()
    plt.plot(record['error'])
    plt.title('Кривая обучения')


# Кривая валидации
def plot_val_curve(record, X_test, Y_test, optimal_a):
    val_score = []
    for w in record['w']:
        val_score.append(func(w, X_test, Y_test, a=optimal_a))
    plt.figure()
    plt.plot(val_score)
    plt.title('Валидационная кривая')


# График зависимости среднеквадратичной ошибки

errors = []
for a in np.linspace(0, 0.1, 10):
    w_optimal, _ = fit_linear(X, Y, a=a)
    error = func(w_optimal, X, Y, a=a)
    errors.append(error)

plt.plot(errors)

# Оптимальное значение α

mean_errors = {}
for a_temp in np.linspace(0, 1, 100):
    record_cv = cross_validation(df, a=a_temp)
    mean_errors[a_temp] = mean_cv_record(record_cv)
min_val = sorted(mean_errors.items(), key=lambda x: x[1])[0]
optimal_a = min_val[0]
print('оптимальное alpha', min_val[0])

X_train, X_test, Y_train, Y_test = train_test_split_df(df)
w, history = fit_linear(X_train, Y_train, a=optimal_a)
plot_learning_curve(history, X_test, Y_test, optimal_a)
plot_val_curve(history, X_test, Y_test, optimal_a)

#test2
