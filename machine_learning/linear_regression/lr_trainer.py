import numpy as num_py
import machine_learning.common_utils.common_helper as common_utils
import linear_regression as lr


class LinearRegressionTrainer:

    def __init__(self, linear_regression):
        """
        Initiate the trainer with linear regression object

        :param linear_regression:
        """
        self.lr_object = linear_regression
        self.alpha = linear_regression.alpha
        self.no_of_iterations = linear_regression.num_iterations

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
        return self.lr_object.gradient_decent(x, Y_train, theta)

    @staticmethod
    def predict(X, theta):
        """
        This method predicts the output for given dataset and trained lr

        :param X:
        :param theta:
        :return:
        """
        return lr.LinearRegression.h_of_theta(X, theta)
