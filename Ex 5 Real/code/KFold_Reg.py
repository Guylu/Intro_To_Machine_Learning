from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

X, y = datasets.load_diabetes(return_X_y=True)
X_train = X[:50]
X_test = X[:51]
y_train = y[:50]
y_test = y[:51]


def regulize():
    ranges = np.arange(0, 20, 0.1)
    n = len(ranges)
    h = [None] * n
    Error_train = [0] * n
    Error_val = [0] * n
    for i, lam in enumerate(ranges):
        kfold = KFold(5, True, 1)
        for train, validation in kfold.split(X_train):
            h[i] = Lasso(alpha=lam)
            h[i].fit(X_train[train], y_train[train])
            Error_train[i] += np.mean(
                (h[i].predict(X_train[train]) - y_train[train]) ** 2)
            Error_val[i] += np.mean(
                (h[i].predict([validation]) - y_train[validation]) ** 2)
    h_star = h[np.argmin(Error_val)]
    plt.plot(Error_train)
    plt.plot(Error_val)
    plt.show()


def regulize2():
    ranges = np.arange(0, 20, 0.1)
    n = len(ranges)
    h = [None] * n
    Error_train = [0] * n
    Error_val = [0] * n
    for i, lam in enumerate(ranges):
        kfold = KFold(5, True, 1)
        for train, validation in kfold.split(X_train):
            h[i] = Ridge(alpha=lam)
            h[i].fit(X_train[train], y_train[train])
            Error_train[i] += np.mean(
                (h[i].predict(X_train[train]) - y_train[train]) ** 2)
            Error_val[i] += np.mean(
                (h[i].predict([validation]) - y_train[validation]) ** 2)
    h_star = h[np.argmin(Error_val)]
    plt.plot(Error_train)
    plt.plot(Error_val)
    plt.show()


regulize()
regulize2()
