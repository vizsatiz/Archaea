import numpy as num_py
import machine_learning.common_utils.common_helper as common_utils
import machine_learning.common_utils.lambda_regularization as lambda_regularization


class LinearRegression(lambda_regularization.Regularization):
    def __init__(self, alpha, no_of_iterations, lambda_regularization):
        """
            Initialize the logistic regression

            :param alpha: learning rate
            :param iterations: number of iterations
            """
        self.alpha = alpha
        self.num_iterations = no_of_iterations
        self.lambda_regularization = lambda_regularization
        super(LinearRegression, self).__init__(lambda_regularization)

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
        hypothesis = LinearRegression.h_of_theta(x, theta)
        abs_error = hypothesis - y
        abs_error_sqrd = num_py.multiply(abs_error, abs_error)
        j_cost = (num_py.sum(abs_error_sqrd)) / float(2 * m)
        return j_cost

    def regularized_cost(self, x, y, theta):
        """
        Method returns regularized cost

        :param x:
        :param y:
        :param theta:
        :return:
        """
        return self.cost(x, y, theta) + self.cost_regularization(theta)

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
        for j_history_index in range(0, self.num_iterations):
            hypothesis = self.h_of_theta(x, theta)
            error = hypothesis - y
            for index in range(0, num_py.size(theta) - 1):
                diff_error = num_py.multiply(error, x[:, index])
                theta = theta - (self.alpha * (num_py.sum(diff_error) + self.gradient_regularization(theta)) / float(m))
            j_history[j_history_index] = self.regularized_cost(x, y, theta)
        return j_history, theta

    @staticmethod
    def h_of_theta(x, theta):
        """
        Method for finding the hypothesis/prediction for given dataset h(theta)

        :param x:
        :param theta:
        :return:
        """
        common_utils.validate_matrix_multiplication_dimensions(x, theta)
        return num_py.dot(x, theta)
