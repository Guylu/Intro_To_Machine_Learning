import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def f(x):
    return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)


def Q_1_b_c(k,sig):
    m = 1500
    X = np.random.uniform(1, -3.2, m).T
    eps = np.random.normal(0, sig, m)
    y = f(X) + eps
    D, T, y_D, y_T = train_test_split(X, y, test_size=500)
    D = D.reshape(-1, 1)
    y_D = y_D.reshape(-1, 1)
    T = T.reshape(-1, 1)
    y_T = y_T.reshape(-1, 1)
    Error_train = [0] * 15
    Error_val = [0] * 15
    h = [None] * 15
    for d in range(15):
        kfold = KFold(k, True, 1)
        for train, validation in kfold.split(D):
            h[d] = make_pipeline(PolynomialFeatures(d), LinearRegression())
            h[d].fit(D[train], y_D[train])
            Error_train[d] += np.mean(
                (h[d].predict(D[train]) - y_D[train]) ** 2)
            Error_val[d] += np.mean(
                (h[d].predict(D[validation]) - y_D[validation]) ** 2)
    plt.plot(Error_train)
    plt.plot(Error_val)
    plt.show()
    h_star = h[np.argmin(Error_val)]
    h_star_test_error = np.mean((h_star.predict(T) - y_T) ** 2)
    return h[np.argmin(Error_val)]


Q_1_b_c(2,1)
Q_1_b_c(5,1)

Q_1_b_c(2,5)
Q_1_b_c(5,5)
