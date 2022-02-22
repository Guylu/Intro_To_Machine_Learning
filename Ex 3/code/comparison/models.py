import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class methods:
    """
    class of different ML algorithms
    """
    model = None
    coeff = np.empty(2)
    bias = 0

    def fit(self, X, y):
        """
        - Given a training set as X P R
        dˆm and y P t˘1u m, this method learns the
        parameters of the model and stores the trained model (namely,
        \the variables that define hypothesis chosen) in self.model. The
        method returns nothing.
        :param X: matrix
        :param y: +-1
        :return: nothing
        """
        pass

    def predict(self, X):
        """
        Given an unlabeled test set X P R
        dˆm1
        , predicts the label of each sample.
        :param X: matrix
        :return:  a vector of predicted labels y P t˘1u m1.
        """
        w = self.coeff
        return np.sign(X.dot(w) + self.bias)

    def score(self, X, y):
        """
         Given an unlabeled test set X P R
        dˆm1
        and the true labels y P t˘1u
        m1
        of this test set, r
        :param X: matrix
        :param y: vector
        :return:
        a dictionary with the following fields:
        • num samples: number of samples in the test set
        • error: error (misclassification) rate
        • accuracy: accuracy
        • FPR: false positive rate
        • TPR: true positive rate
        • precision: precision
        • recall: recall
        """
        # dict to hold final result
        dict = {}
        self.fit(X, y)
        y_hat = self.predict(X)
        # parameters
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        # calculating parameters:
        for i in range(len(y_hat)):
            if y[i] == y_hat[i] == 1:
                TP += 1
            if y_hat[i] == 1 and y[i] != y_hat[i]:
                FP += 1
            if y[i] == y_hat[i] == 0:
                TN += 1
            if y_hat[i] == 0 and y[i] != y_hat[i]:
                FN += 1

        # attributes asked:
        dict["num_samples"] = X.shape[1]
        dict["error"] = np.sum(y != y_hat)
        dict["accuracy"] = np.sum(y == y_hat) / (len(y))
        # handling division by 0
        dict["FPR"] = FN / (FN + TP) if (FN + TP) != 0 else 1
        dict["TPR"] = TP / (TP + FN) if (TP + FN) != 0 else 1
        dict["precision"] = TP / (TP + FP) if (TP + FP) != 0 else 1
        dict["recall"] = TP / (TP + FN) if (TP + FN) != 0 else 1
        return dict


class Perceptron(methods):
    """
    Implement a half-space classifier using the perceptron algorithm
    """

    def fit(self, X, y):
        """
        perception code
        :param X:
        :param y:
        :return:
        """
        X = np.insert(X, 0, 1, axis=1)
        w = np.zeros(X.shape[1])
        while not (np.sign(X.dot(w)) == y).all():
            for i in range(len(X)):
                if (np.dot(w, X[i]) * y[i]) <= 0:
                    w = w + X[i] * y[i]
        self.model = w
        self.coeff = w[0:2]
        self.bias = w[2]


class LDA(methods):
    """
    Implement the LDA classifier from Question 1.
    """
    # lambda function for calculating LDA
    delta = None

    def fit(self, X, y):
        """
        LDA code
        :param X:
        :param y:
        :return:
        """
        m = len(y)
        # num of 1's and -1's
        ones = np.sum(y[np.where(y == 1)])
        minus_ones = -np.sum(y[np.where(y == -1)])
        # estimation for pr=1, pr=-1
        pr_y_1 = ones / len(y)
        pr_y_minus_1 = 1 - pr_y_1
        # estimation expectency:
        mu_1 = (np.sum(X[np.where(y == 1)], axis=0) / ones).reshape(2, 1)
        mu_minus_1 = (np.sum(X[np.where(y == -1)], axis=0) /
                      minus_ones).reshape(2, 1)
        # putting all vars in dicts
        mus = {-1: mu_minus_1, 1: mu_1}
        pr_y = {1: pr_y_1, -1: pr_y_minus_1}
        sigma = np.zeros(shape=(2, 2))
        # calculating cov matrix:
        for mu in [-1, 1]:
            for xi in X[np.where(y == mu)]:
                sigma += np.dot((xi - mus[mu]), (xi - mus[mu]).T)
        sigma /= m - 2
        sigma_inv = np.linalg.inv(sigma)
        # defining lambda:
        self.delta = lambda x, i: np.dot(np.dot(x.T, sigma_inv), mus[i]) - \
                                  0.5 * \
                                  np.dot(np.dot(mus[i].T, sigma_inv), mus[i]) \
                                  + np.log(pr_y[i])

    def predict(self, X):
        """
        prediction as in PDF
        :param X:
        :return:
        """
        pred = np.empty(X.shape[0])
        for i, x_i in enumerate(X):
            pred[i] = 1 if self.delta(x_i, 1) > self.delta(x_i, -1) else -1
        return pred


class SVM(methods):
    """
    Implement SVM.
    """

    def fit(self, X, y):
        self.model = SVC(C=1e10, kernel="linear")
        self.model.fit(X, y)
        self.coeff = self.model.coef_
        self.bias = self.model.intercept_


class Logistic(methods):
    """
    - Implement logistic regression.
    """

    def fit(self, X, y):
        self.model = LogisticRegression(solver="liblinear")
        self.model.fit(X, y)
        self.coeff = self.model.coef_
        self.bias = self.model.intercept_


class DecisionTree(methods):
    """
    Implement a decision tree
    """

    def fit(self, X, y):
        self.model = DecisionTreeClassifier(random_state=0)
        self.model.fit(X, y)
