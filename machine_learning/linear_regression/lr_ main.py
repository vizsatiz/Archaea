import numpy as num_py
import machine_learning.common_utils.common_helper as common_utils
import machine_learning.common_utils.error_messages as ERROR


class CostAndGradient:
    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.num_iterations = iterations

    def cost(self, x, y, theta):
        m = num_py.size(y)
        self.validated_x_y_theta_dimensions(x, y, theta)
        hypothesis = self.h_of_theta(x, theta)
        abs_error = hypothesis - y
        abs_error_sqrd = num_py.multiply(abs_error, abs_error)
        j_cost = (num_py.sum(abs_error_sqrd)) / (2 * m)
        return j_cost

    def gradient_decent(self, x, y, theta):
        m = num_py.size(y)
        self.validated_x_y_theta_dimensions(x, y, theta)
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
    def validated_x_y_theta_dimensions(x, y, theta):
        shape_of_x = num_py.shape(x)
        number_of_rows_of_x = shape_of_x[0]
        number_of_columns_of_x = shape_of_x[1]
        shape_of_y = num_py.shape(y)
        number_of_rows_of_y = shape_of_y[0]
        number_of_columns_of_y = shape_of_y[1]
        shape_of_theta = num_py.shape(theta)
        number_of_rows_of_theta = shape_of_theta[0]
        number_of_columns_of_theta = shape_of_theta[1]
        if (number_of_rows_of_theta != number_of_columns_of_x) & (number_of_columns_of_theta != 1):
            raise ValueError(ERROR.MATRIX_DIMENSION_MISMATCH_ERROR)
        if (number_of_rows_of_x != number_of_rows_of_y) & (number_of_columns_of_y != 1):
            raise ValueError(ERROR.MATRIX_DIMENSION_MISMATCH_ERROR)

    @staticmethod
    def h_of_theta(x, theta):
        return num_py.dot(x, theta)

    @staticmethod
    def train_linear_regression(x, y, no_of_iterations, alpha):
        m = num_py.size(y)
        x = common_utils.add_column_of_ones(x, m)
        number_of_features = x.shape[1]
        # no of elements in theta
        theta = num_py.zeros((number_of_features, 1))
        linear_regression = CostAndGradient(alpha, no_of_iterations)
        return linear_regression.gradient_decent(x, y, theta, alpha, no_of_iterations)
