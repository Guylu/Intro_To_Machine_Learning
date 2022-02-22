"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

        # func to transform weighs
        self.transformation = \
            np.vectorize(lambda d_t, w_t, y_i, h_t, sum: \
                             np.divide((d_t * np.exp(-w_t * y_i * h_t)), sum))

    def train(self, X, y):
        """
        Parameters ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples) Train this classifier over the sample
        (X,y) After finish the training return the weights of the samples in
        the last iteration.
        """
        # start with uniform weights
        D = np.ones(y.shape) / y.shape
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            pred = self.h[t].predict(X)
            epsilon = np.sum(D[pred != y])
            self.w[t] = 0.5 * np.log((1 / epsilon) - 1)
            # updating weights
            D = self.transformation(D, self.w[t], y, pred, np.sum(D))
        return D

    def predict(self, X, max_t):
        """
        Parameters ---------- X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for
        the classification
        :return: y_hat : a prediction vector for X.
        shape=(num_samples) Predict only with max_t weak learners,
        """
        # func to give weighted predictions
        weighted = lambda h, w: h.predict(X) * w
        # summing the weighted predictions and returns the sign
        return np.sign(np.sum(np.vectorize(weighted, otypes=[object])
                              (self.h[:max_t], self.w[:max_t])))

    def error(self, X, y, max_t):
        """
        Parameters ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the
        number of classifiers to use for the classification
        :return: error :
        the ratio of the correct predictions when predict only with max_t
        weak learners (float)
        """
        return np.mean(self.predict(X, max_t) != y)



##################################
# my implementation is in ex4_tools :)
