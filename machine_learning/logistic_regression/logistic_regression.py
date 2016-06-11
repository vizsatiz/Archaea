import numpy as num_py
import machine_learning.common_utils.common_helper as common_utils
import machine_learning.common_utils.lambda_regularization as lambda_regularization


class LogisticRegression(lambda_regularization.Regularization):
    def __init__(self, alpha, iterations, regression_function):
        """
        Initialize the logistic regression

        :param alpha: learning rate
        :param iterations: number of iterations
        :param regression_function: The regression function
        """
        self.alpha = alpha
        self.num_iterations = iterations
        self.regression_function = regression_function
        super(LogisticRegression, self).__init__(lambda_regularization)

    def cost(self, x, y, theta):
        """
            This method calculates the cost of the predicted output by the linear regression.
            This is the error we are trying to minimize

            :param x:
            :param y:
            :param theta:
            :return:
            """
        m = num_py.size(y)
        common_utils.validated_x_y_theta_dimensions(x, y, theta)
        hypothesis = self.h_of_theta(x, theta)
        log_h_of_x = num_py.log(hypothesis)
        log_one_minus_h_of_x = num_py.log(1 - hypothesis)
        term_1 = num_py.multiply(y, log_h_of_x)
        term_2 = num_py.multiply(y, log_one_minus_h_of_x)
        sum_of_terms = term_1 + term_2
        j_cost = (num_py.sum(sum_of_terms)) / (-1 * m)
        return j_cost

    def regularized_cost(self, x, y, theta):
        """
        This method calculates the cost of the predicted output by the linear regression.
        This is the error we are trying to minimize

        :param x:
        :param y:
        :param theta:
        :return:
        """
        return self.cost(x, y, theta)

    def gradient_decent(self, x, y, theta):
        """
            Gradient Descent is the derivative of cost function which is the rate of change of error
            We should find the Theta that minimises this GD value

            :param x:
            :param y:
            :param theta:
            :return:
            """
        m = num_py.size(y)
        common_utils.validated_x_y_theta_dimensions(x, y, theta)
        j_history = num_py.zeros((self.num_iterations, 1))
        for j_history_index in range(1, self.num_iterations):
            hypothesis = self.h_of_theta(x, theta)
            error = hypothesis - y
            for index in range(0, num_py.size(theta) - 1):
                diff_error = num_py.multiply(error, x[:, index])
                theta = theta - (self.alpha * (num_py.sum(diff_error) + self.gradient_regularization(theta)) / m)
            j_history[j_history_index] = self.cost(x, y, theta)
        return j_history, theta

    def h_of_theta(self, x, theta):
        """
            Method for finding the hypothesis/prediction for given dataset h(theta)

            :param x:
            :param theta:
            :return:
            """
        return self.regression_function(x)
