import numpy as num_py


class RidgeRegressionTrainer:

    def __init__(self, ridge_regression, degree):
        """
        Initiate the trainer with linear regression object

        :param linear_regression:
        """
        self.ridge_reg = ridge_regression
        self.degree = degree

    def train(self, X_train, Y_train):
        """
        This function takes the training data 1. Adds the bais colomns to X_train
        Also trains the Linear Regression

        :param X_train: Training data value
        :param Y_train: Results for supervised learning
        :return:        """

        self.ridge_reg.fit(num_py.vander(X_train, self.degree + 1), Y_train)

    def predict(self, X):
        """
        This method predicts the output for given dataset and trained lr

        :param X:
        :return:
        """
        return self.ridge_reg.predict(num_py.vander(X, self.degree + 1))

    def error_and_variance(self, X_test, Y_test):
        """
        This function returns the error on given test data

        :param X_test:
        :param Y_test:
        :return:
        """
        mean_error = num_py.mean(self.ridge_reg.predict(X_test - Y_test) ** 2)
        variance = self.ridge_reg.score(X_test, Y_test)
        return mean_error, variance

    def regression_coefficients(self):
        """
        Current state of coefficients

        :return:
        """
        return self.ridge_reg.coef_
