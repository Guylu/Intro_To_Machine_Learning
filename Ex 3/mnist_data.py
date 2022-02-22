import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

# PDF code:
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_images = np.logical_or((y_train == 0), (y_train == 1))
test_images = np.logical_or((y_test == 0), (y_test == 1))
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]


def q_12():
    """
    question 12
    :return:
    """
    zeros = x_test[np.where(y_test == 0)]
    plt.imshow(zeros[0])
    plt.show()
    plt.imshow(zeros[1])
    plt.show()
    plt.imshow(zeros[2])
    plt.show()

    ones = x_train[np.where(y_train == 1)]
    plt.imshow(ones[0])
    plt.show()
    plt.imshow(ones[1])
    plt.show()
    plt.imshow(ones[2])
    plt.show()


def rearrange_data(X):
    """
    rearranges 3d tensor to a matrix
    :param X:3d tensor
    :return:2d matrix
    """
    mat = np.empty(shape=(X.shape[0], X.shape[1] * X.shape[2]))
    for i, cell in enumerate(mat):
        mat[i] = X[i].flatten()
    return mat


def q_14():
    """
    question 14
    :return:
    """
    M = np.array([50, 100, 300, 500])
    k = 50
    # lists to hold accuracies
    logistical_list = []
    svc_list = []
    dec_tree_list = []
    k_neighbors_list = []
    for m in M:
        logistical_time = 0
        svc_time = 0
        dec_tree_time = 0
        k_neighbors_time = 0
        logistical_accuracy = 0
        svc_accuracy = 0
        dec_tree_accuracy = 0
        k_neighbors_accuracy = 0
        for i in range(1, k + 1):
            b = np.random.randint(0, x_train.shape[0], m)
            X_train = x_train[b]
            Y_train = y_train[b]
            # need to redraw:
            while len(np.unique(Y_train)) == 1:
                b = np.random.randint(0, x_train.shape[0], m)
                X_train = x_train[b]
                Y_train = y_train[b]
            X_train = rearrange_data(X_train)
            X_test = rearrange_data(x_test)

            # all models fitting and predictions:

            start = time.time()
            logistical = LogisticRegression()
            logistical.fit(X_train, Y_train)
            y_hat_log = logistical.predict(X_test)
            end = time.time()
            logistical_time += end - start

            start = time.time()
            svc = SVC()
            svc.fit(X_train, Y_train)
            y_hat_svc = svc.predict(X_test)
            end = time.time()
            svc_time += end - start

            start = time.time()
            dec_tree = DecisionTreeClassifier()
            dec_tree.fit(X_train, Y_train)
            y_hat_tree = dec_tree.predict(X_test)
            end = time.time()
            dec_tree_time += end - start

            start = time.time()
            k_neighbors = KNeighborsClassifier()
            k_neighbors.fit(X_train, Y_train)
            y_hat_k_neighbors = k_neighbors.predict(X_test)
            end = time.time()
            k_neighbors_time += end - start

            # accuracies:
            logistical_accuracy += (
                    np.sum(y_test == y_hat_log) / (len(y_test)))
            svc_accuracy += (
                    np.sum(y_test == y_hat_svc) / (len(y_test)))
            dec_tree_accuracy += (
                    np.sum(y_test == y_hat_tree) / (len(y_test)))
            k_neighbors_accuracy += (
                    np.sum(y_test == y_hat_k_neighbors) / (len(y_test)))

        logistical_list.append(logistical_accuracy / k)
        svc_list.append(svc_accuracy / k)
        dec_tree_list.append(dec_tree_accuracy / k)
        k_neighbors_list.append(k_neighbors_accuracy / k)

        a = plt.scatter([m], [logistical_accuracy / k], label="logistical",
                        c="blue")
        b = plt.scatter([m], [svc_accuracy / k], label="SVC", c="red")
        c = plt.scatter([m], [dec_tree_accuracy / k], label="dec_tree",
                        c="purple")
        d = plt.scatter([m], [k_neighbors_accuracy / k], label="k_neighbors",
                        c="green")
        plt.legend(handles=[a, b, c, d])
        print("logistical_time: " + str(logistical_time) + " on m: " + str(m))
        print("svc_time: " + str(svc_time) + " on m: " + str(m))
        print("dec_tree_time: " + str(dec_tree_time) + " on m: " + str(m))
        print("k_neighbors_time: " + str(k_neighbors_time) + " on m: " + str(m))
        print()
    plt.title("Graph 2: Accuracy per training size")
    plt.xlabel("size of training data")
    plt.ylabel("accuracy")

    plt.plot(M, logistical_list, c="blue")
    plt.plot(M, svc_list, c="red")
    plt.plot(M, dec_tree_list, c="purple")
    plt.plot(M, k_neighbors_list, c="green")
    plt.show()


# calling func
q_12()
q_14()
