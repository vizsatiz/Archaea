import numpy as num_py
import machine_learning.common_utils.common_helper as common_utils
import machine_learning.logistic_regression.regression_function as reg_functions


class CostAndGradient:
    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.num_iterations = iterations

    def cost(self, x, y, theta):
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

    def gradient_decent(self, x, y, theta):
        m = num_py.size(y)
        common_utils.validated_x_y_theta_dimensions(x, y, theta)
        j_history = num_py.zeros((self.num_iterations, 1))
        for j_history_index in range(1, self.num_iterations):
            hypothesis = self.h_of_theta(x, theta)
            error = hypothesis - y
            for index in range(0, num_py.size(theta) - 1):
                diff_error = num_py.multiply(error, x[:, index])
                theta = theta - (self.alpha * (num_py.sum(diff_error)) / m)
            j_history[j_history_index] = self.cost(x, y, theta)
        return j_history, theta

    @staticmethod
    def train_logistic_regression(self, x, y, no_of_iterations, alpha):
        m = num_py.size(y)
        x = common_utils.add_column_of_ones(x, m)
        number_of_features = x.shape[1]
        # no of elements in theta
        theta = num_py.zeros((number_of_features, 1))
        logistic_regression = CostAndGradient(alpha, no_of_iterations)
        return logistic_regression.gradient_decent(x, y, theta)

    @staticmethod
    def h_of_theta(x, theta):
        return reg_functions.sigmoid(x)

    @staticmethod
    def predict(x, theta, threshold):
        m, n = x.shape
        p = num_py.zeros(shape=(m, 1))
        h = reg_functions.sigmoid(x.dot(theta.T))
        for it in range(0, h.shape[0]):
            if h[it] > threshold:
                p[it, 0] = 1
            else:
                p[it, 0] = 0
        return p
