import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import models


def draw_points(m):
    """
    given an integer m returns a pair X, y where X is
    2 ˆ m matrix where each column represents an i.i.d sample
    from the distribution above,
    and y P t˘1u m is its corresponding label, according to fpxq.
    :param m: number
    :return:
    """
    X = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=m)
    y = np.empty(m)
    w = np.array([0.3, -0.5])
    for i, x in enumerate(X):
        y[i] = np.sign(np.dot(w, x) + 0.1)
    return X, y


def f_x(t):
    """
    calculating real plane
    :param t:
    :return:
    """
    return ((-t * 0.3) - 0.1) / -0.5


def question_9():
    """
    question 9 in PDF
    :return:
    """
    M = np.array([5, 10, 15, 25, 70])
    for m in M:
        # drawing points
        X, y = draw_points(m)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        points = np.linspace(-5, 5, 3)
        res1 = np.apply_along_axis(f_x, 0, points.T)
        plt.title(
            "Classified Points and separating hyper planes with " + str(m) + " "
                                                                             "points")
        plt.xlabel("x")
        plt.ylabel("y")
        y_one = X[np.where(y == 1)]
        y_minus_one = X[np.where(y == -1)]
        plt.scatter(y_one[:, 0], y_one[:, 1], c="blue",
                    label="classified 1")

        plt.scatter(y_minus_one[:, 0], y_minus_one[:, 1],
                    c="red",
                    label="classified -1")
        # real plane:
        plt.plot(points, res1, label="real", c="black")

        # perceptron model:
        perceptron = models.Perceptron()
        perceptron.fit(X, y)
        w = perceptron.model

        # perceptron plane
        w_x = lambda t: ((-t * w[1]) - w[0]) / w[2]
        points2 = np.linspace(-5, 5, 3)
        res2 = np.apply_along_axis(w_x, 0, points2)

        plt.plot(points2, res2, label="perceptron", c="orange")
        # svm model:
        svm = models.SVM()
        svm.fit(X, y)
        w[1] = svm.model.coef_[0][0]
        w[2] = svm.model.coef_[0][1]
        w[0] = svm.model.intercept_[0]

        w_x = lambda t: ((-t * w[1]) - w[0]) / w[2]
        points3 = np.linspace(-5, 5, 3)
        res3 = np.apply_along_axis(w_x, 0, points3)

        # svm plane:
        plt.plot(points3, res3, label="SVM", c="blue")

        plt.legend()
        plt.show()


def question_10():
    """
    question 10
    :return:
    """
    M = np.array([5, 10, 15, 25, 70])
    k = 10000
    N = 500
    # lists to hold accuracies
    perceptron_list = []
    svm_list = []
    lda_list = []
    for m in M:
        # accuracies:
        perceptron_accuracy = 0
        svm_accuracy = 0
        lda_accuracy = 0
        for i in range(1, N + 1):
            X_train, y_train = draw_points(m)
            # if need to redraw:
            while len(np.unique(y_train)) == 1:
                X_train, y_train = draw_points(m)
            X_test, y_test = draw_points(k)
            # perceptron:
            perceptron = models.Perceptron()
            perceptron.fit(X_train, y_train)
            w_perc = perceptron.model
            # svm:
            svm = models.SVM()
            svm.fit(X_train, y_train)
            w_svm = np.empty(3)
            w_svm[1] = svm.model.coef_[0][0]
            w_svm[2] = svm.model.coef_[0][1]
            w_svm[0] = svm.model.intercept_[0]
            # LDA:
            lda = models.LDA()
            lda.fit(X_train, y_train)
            y_hat_lda = lda.predict(X_test)
            # calculating y_hat for each model
            y_hat_perc = np.empty(k)
            for j, x in enumerate(X_test):
                y_hat_perc[j] = np.sign(
                    np.dot(np.array([w_perc[1], w_perc[2]]), x) + w_perc[0])
            y_hat_svm = np.empty(k)
            for j, x in enumerate(X_test):
                y_hat_svm[j] = np.sign(np.dot(np.array([w_svm[1], w_svm[2]]),
                                              x) + w_svm[0])

            perceptron_accuracy += (
                    np.sum(y_test == y_hat_perc) / (len(y_test)))
            svm_accuracy += (np.sum(y_test == y_hat_svm) / (len(y_test)))
            lda_accuracy += (np.sum(y_test == y_hat_lda) / (len(y_test)))

        # plotting:

        perceptron_list.append(perceptron_accuracy / N)
        svm_list.append(svm_accuracy / N)
        lda_list.append(lda_accuracy / N)

        a = plt.scatter([m], [perceptron_accuracy / N], label="perceptron",
                        c="blue")
        b = plt.scatter([m], [svm_accuracy / N], label="SVM", c="red")
        c = plt.scatter([m], [lda_accuracy / N], label="LDA", c="purple")
        plt.legend(handles=[a, b, c])
    plt.title("Graph 1: Accuracy per training size")
    plt.xlabel("size of training data")
    plt.ylabel("accuracy")

    plt.plot(M, perceptron_list, c="blue")
    plt.plot(M, svm_list, c="red")
    plt.plot(M, lda_list, c="purple")

    plt.show()


# calling funcs

question_9()
question_10()
