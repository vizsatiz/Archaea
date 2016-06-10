import numpy as num_py
import machine_learning.common_utils.common_helper as common_utils
import linear_regression as lr


class linear_regression_trainer:
    def __init__(self, linear_regression, alpha, no_of_iterations):
        self.lr_object = linear_regression
        self.alpha = alpha
        self.no_of_iterations = no_of_iterations

    def train(self, X_train, Y_train):
        """
        This function takes the training data 1. Adds the bais colomns to X_train
        Also trains the Linear Regression

        :param X_train: Training data value
        :param Y_train: Results for supervised learning
        :return:
        """
        m = num_py.size(Y_train)
        x = common_utils.add_column_of_ones(X_train, m)
        number_of_features = x.shape[1]
        # no of elements in theta
        theta = num_py.zeros((number_of_features, 1))
        linear_regression = lr.LinearRegression(self.alpha, self.no_of_iterations)
        return linear_regression.gradient_decent(x, Y_train, theta)

    def perdict(self, X, theta):
        """
        This method predicts the output for given dataset and trained lr

        :param X:
        :param theta:
        :return:
        """
        return lr.LinearRegression.h_of_theta(X, theta)
