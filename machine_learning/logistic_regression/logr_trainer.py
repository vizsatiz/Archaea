import numpy as num_py
import logistic_regression as lr
import regression_function as reg_functions
import machine_learning.common_utils.common_helper as common_utils


class logr_trainer:
    def __init__(self, logistic_regression, alpha, no_of_iterations, threshold):
        self.logr_object = logistic_regression
        self.alpha = alpha
        self.no_of_iterations = no_of_iterations
        self.threshold = threshold

    def train(self, X_train, Y_train):
        """
        Train the logistic regression

        :param X_train:
        :param Y_train:
        :return:
        """
        m = num_py.size(Y_train)
        x = common_utils.add_column_of_ones(X_train, m)
        number_of_features = x.shape[1]
        # no of elements in theta
        theta = num_py.zeros((number_of_features, 1))
        logistic_regression = lr.LogisticRegression(self.alpha, self.no_of_iterations)
        return logistic_regression.gradient_decent(x, Y_train, theta)

    def predict(self, X, theta):
        """
        Predict the output of prediction with threshold set in __init__

        :param X:
        :param theta:
        :return:
        """
        m, n = X.shape
        p = num_py.zeros(shape=(m, 1))
        h = reg_functions.sigmoid(X.dot(theta.T))
        for it in range(0, h.shape[0]):
            if h[it] > self.threshold:
                p[it, 0] = 1
            else:
                p[it, 0] = 0
        return p