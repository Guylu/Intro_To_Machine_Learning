import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class methods:
    model = None

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
        pass

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
        pass


class Perceptron(methods):
    """
    Implement a half-space classifier using the perceptron algorithm
    """

    def fit(self, X, y):
        w = np.zeros(X.shape[1])
        t = 1
        while X.dot(w) != y:
            for i, x in enumerate(X):
                if (np.dot(w, X[i]) * y[i]) <= 0:
                    w = w + X[i] * y[i]
            t += 1
        self.model = w

    def predict(self, X):
        w = self.model
        return X.dot(w)

    def score(self, X, y):
        dict = {}
        self.fit(X, y)
        w = self.model
        y_hat = X.dot(w)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(y_hat)):
            if y[i] == y_hat[i] == 1:
                TP += 1
            if y_hat[i] == 1 and y[i] != y_hat[i]:
                FP += 1
            if y[i] == y_hat[i] == 0:
                TN += 1
            if y_hat[i] == 0 and y[i] != y_hat[i]:
                FN += 1

        dict["num_samples"] = X.shape[1]
        dict["error"] = np.sum(y != y_hat)
        dict["accuracy"] = np.sum(y == y_hat) / (len(y))
        dict["FPR"] = FP
        dict["TPR"] = TP
        dict["precision"] = TP / (TP + FP)
        dict["recall"] = TP / (TP + FN)

    class LDA(methods):
        """
        Implement the LDA classifier from Question 1.
        """

        def fit(self, X, y):
            w = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                # w[i] = 1 if else -1
                pass

        def predict(self, X):
            pass

        def score(self, X, y):
            pass

    class SVM(methods):
        """
        Implement SVM.
        """

        def fit(self, X, y):
            self.model = SVC(C=1e10, kernel="linear")
            self.model.fit(X, y)

        def predict(self, X):
            return self.model.predict(X)

        def score(self, X, y):
            return self.score(X, y)

    class Logistic(methods):
        """
        - Implement logistic regression.
        """

        def fit(self, X, y):
            self.model = LogisticRegression(solver="liblinear")
            self.model.fit(X, y)

        def predict(self, X):
            return self.model.predict(X)

        def score(self, X, y):
            return self.score(X, y)

    class DecisionTree(methods):
        """
        Implement a decision tree
        """

        def fit(self, X, y):
            self.model = DecisionTreeClassifier(random_state=0)
            self.model.fit(X, y)

        def predict(self, X):
            return self.model.predict(X)

        def score(self, X, y):
            return self.score(X, y)
