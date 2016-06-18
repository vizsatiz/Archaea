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
