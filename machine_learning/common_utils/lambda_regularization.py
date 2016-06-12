import numpy as num_py


class Regularization(object):
    def __init__(self, lambda_param):
        self.lambda_param = lambda_param

    def cost_regularization(self, theta):
        # type: (object) -> object
        """
        Calculates the regression for cost function  lambda * sigma(theta squared)

        :param theta:
        :return:
        """
        theta_squared = num_py.multiply(theta, theta)
        sum_of_theta = num_py.sum(theta_squared)
        return self.lambda_param * sum_of_theta

    def gradient_regularization(self, theta):
        """
        Returns gradient descent lambda_param * theta

        :param theta:
        :return:
        """
        return self.lambda_param * theta

