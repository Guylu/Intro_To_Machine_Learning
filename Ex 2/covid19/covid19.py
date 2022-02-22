import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split


def fit_linear_regression(X, y):
    """
    The function returns two sets of values:
    the first is a numpy array of the coefficients vector ‘w‘ (think what
    should its dimension be)
    the second is a numpy array of the singular values of X.
    :param X:(numpy array with p rows and n columns
    :param y:numpy array with n rows
    :return:
    """
    singular_vals = np.linalg.svd(X, compute_uv=False)
    X_dagger_T = np.linalg.pinv(X)
    w = X_dagger_T.dot(y)
    return w, singular_vals


def load_data(path):
    """
    load the data
    :param path: path of data
    :return: refurbished data
    """
    h = pd.read_csv(path)
    detected = h['detected']
    log_dec = np.log(detected)
    h["log_detected"] = log_dec
    h['date'] = h["date"].str.replace("/", "")
    return h.values.astype('float64')


X = load_data(r"C:\University\Year 2\Semester 2\67577 Intro To Machine "
              "Learning\Ex's\Ex 2\covid19/covid19_israel.csv")

real_loged = X[:, 3]
real = X[:, 2]
num_of_day = X[:, 0:1]
# just get the w hat
w = fit_linear_regression(num_of_day, real_loged)[0]

y_hat = num_of_day.dot(w)
y_hat_exp = np.exp(y_hat)

plt.scatter(num_of_day, real_loged)
plt.plot(num_of_day, y_hat)
plt.title("Q21 - fitting COVID 19 patients logged ")
plt.xlabel("number of day")
plt.ylabel("number of infected")
plt.show()

plt.scatter(num_of_day, real)
plt.plot(num_of_day, y_hat_exp)
plt.title("Q21 - fitting COVID 19 patients ")
plt.xlabel("number of day")
plt.ylabel("number of infected")

plt.show()
