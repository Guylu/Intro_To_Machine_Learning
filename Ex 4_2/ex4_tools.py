"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

This module provides some useful tools for Ex4.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import product
from matplotlib.pyplot import imread
import os
from sklearn.model_selection import train_test_split
import adaboost


def find_threshold(D, X, y, sign, j):
    """
    Finds the best threshold.
    D =  distribution
    S = (X, y) the data
    """
    # sort the data so that x1 <= x2 <= ... <= xm
    sort_idx = np.argsort(X[:, j])
    X, y, D = X[sort_idx], y[sort_idx], D[sort_idx]

    thetas = np.concatenate([[-np.inf], (X[1:, j] + X[:-1, j]) / 2, [np.inf]])
    minimal_theta_loss = np.sum(
        D[y == sign])  # loss of the smallest possible theta
    losses = np.append(minimal_theta_loss,
                       minimal_theta_loss - np.cumsum(D * (y * sign)))
    min_loss_idx = np.argmin(losses)
    return losses[min_loss_idx], thetas[min_loss_idx]


class DecisionStump(object):
    """
    Decision stump classifier for 2D samples
    """

    def __init__(self, D, X, y):
        self.theta = 0
        self.j = 0
        self.sign = 0
        self.train(D, X, y)

    def train(self, D, X, y):
        """
        Train the classifier over the sample (X,y) w.r.t. the weights D over X
        Parameters
        ----------
        D : weights over the sample
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        """
        loss_star, theta_star = np.inf, np.inf
        for sign, j in product([-1, 1], range(X.shape[1])):
            loss, theta = find_threshold(D, X, y, sign, j)
            if loss < loss_star:
                self.sign, self.theta, self.j = sign, theta, j
                loss_star = loss

    def predict(self, X):
        """
        Parameters
        ----------
        X : shape=(num_samples, num_features)
        Returns
        -------
        y_hat : a prediction vector for X shape=(num_samples)
        """
        y_hat = self.sign * ((X[:, self.j] <= self.theta) * 2 - 1)
        return y_hat


def decision_boundaries(classifier, X, y, num_classifiers=1, weights=None):
    """
    Plot the decision boundaries of a binary classfiers over X \subseteq R^2

    Parameters
    ----------
    classifier : a binary classifier, implements classifier.predict(X)
    X : samples, shape=(num_samples, 2)
    y : labels, shape=(num_samples)
    title_str : optional title
    weights : weights for plotting X
    :param weights:
    :param classifier:
    :param num_classifiers:
    """
    cm = ListedColormap(['#AAAAFF', '#FFAAAA'])
    cm_bright = ListedColormap(['#0000FF', '#FF0000'])
    h = .003  # step size in the mesh
    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
    y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()], num_classifiers)
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cm)
    # Plot also the training points
    if weights is not None:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=weights, cmap=cm_bright)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks([])
    plt.yticks([])
    plt.title(f'num classifiers = {num_classifiers}')
    plt.draw()


def generate_data(num_samples, noise_ratio):
    """
    generate samples X with shape: (num_samples, 2) and labels y with shape (
    num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    """
    X = np.random.rand(num_samples, 2) * 2 - 1
    radius = 0.5 ** 2
    in_circle = np.sum(X ** 2, axis=1) < radius
    y = np.ones(num_samples)
    y[in_circle] = -1
    y[np.random.choice(num_samples, int(noise_ratio * num_samples))] *= -1

    return X, y


def Q_10__13(noise_ratio=0.0):
    """
    function to display answers for questions 10 - 13
    :param noise_ratio: the noise added to the sample data
    :return: nothing
    """
    # Q_10:
    train_error = np.zeros(500)
    test_error = np.zeros(500)
    # generating data
    train_sample_no_noise, y_train = generate_data(5000, noise_ratio)
    model = adaboost.AdaBoost(DecisionStump, 500)
    # creating model
    D = model.train(train_sample_no_noise, y_train)
    test_sample_no_noise, y_test = generate_data(200, noise_ratio)
    # drawing graph
    for t in range(500):
        train_error[t] = model.error(train_sample_no_noise, y_train, t + 1)
        test_error[t] = model.error(test_sample_no_noise, y_test, t + 1)

    plt.title("Graph for Q10: Training & Test Error With Noise_Ratio= "
              "" + str(noise_ratio))
    plt.xlabel("Number of Classifiers")
    plt.ylabel("Error")
    plt.plot(train_error, label="train")
    plt.plot(test_error, label="test")
    plt.legend()
    plt.show()

    # Q_11:
    T = [5, 10, 50, 100, 200, 500]
    for i, t in enumerate(T):
        plt.subplot(2, 3, i + 1)
        decision_boundaries(model, test_sample_no_noise, y_test, t)

    # Q_11 results:
    plt.show()

    # Q_12:
    min_error = np.argmin(test_error[1:])

    # Q_12 results:
    decision_boundaries(model, train_sample_no_noise, y_train, min_error)
    plt.title("Graph for Q12: Decision boundaries for Num of Classifiers\n "
              "That Minimizes Test Error")
    plt.show()

    # Q_13:
    decision_boundaries(model, train_sample_no_noise, y_train, min_error,
                        weights=D)
    plt.title("Graph for Q13_a: Decision boundaries With Original Weights")
    plt.show()

    decision_boundaries(model, train_sample_no_noise, y_train, min_error,
                        weights=((D / np.max(D)) * 10))
    plt.title("Graph for Q13_b: Decision boundaries With Normalized Weights")
    plt.show()


Q_10__13(0)
Q_10__13(0.01)
Q_10__13(0.4)
