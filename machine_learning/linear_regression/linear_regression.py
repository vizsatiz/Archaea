import numpy as num_py
import machine_learning.common_utils.common_helper as common_utils


class CostAndGradient:
    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.num_iterations = iterations

    def cost(self, x, y, theta):
        m = num_py.size(y)
        common_utils.validated_x_y_theta_dimensions(x, y, theta)
        hypothesis = common_utils.h_of_theta(x, theta)
        abs_error = hypothesis - y
        abs_error_sqrd = num_py.multiply(abs_error, abs_error)
        j_cost = (num_py.sum(abs_error_sqrd)) / (2 * m)
        return j_cost

    def gradient_decent(self, x, y, theta):
        m = num_py.size(y)
        common_utils.validated_x_y_theta_dimensions(x, y, theta)
        j_history = num_py.zeros((self.num_iterations, 1))
        for j_history_index in range(1, self.num_iterations):
            hypothesis = common_utils.h_of_theta(x, theta)
            error = hypothesis - y
            for index in range(0, num_py.size(theta) - 1):
                diff_error = num_py.multiply(error, x[:, index])
                theta = theta - (self.alpha * (num_py.sum(diff_error)) / m)
            j_history[j_history_index] = self.cost(x, y, theta)
        return j_history, theta

    @staticmethod
    def train_linear_regression(x, y, no_of_iterations, alpha):
        m = num_py.size(y)
        x = common_utils.add_column_of_ones(x, m)
        number_of_features = x.shape[1]
        # no of elements in theta
        theta = num_py.zeros((number_of_features, 1))
        linear_regression = CostAndGradient(alpha, no_of_iterations)
        return linear_regression.gradient_decent(x, y, theta)
