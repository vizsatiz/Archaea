import machine_learning.linear_regression.linear_regression as lr
import lr_test_data as test_data
import numpy as num_py
import unittest


class TestLinearRegressionGradientDescent(unittest.TestCase):

    def test_gradient_descent_no_iterations(self):
        alpha = 0
        lambda_param = 0.001
        no_of_iterations = 0
        lr_object = lr.LinearRegression(alpha, no_of_iterations, lambda_param)
        [j_history, theta] = lr_object.gradient_decent(test_data.ZERO_ERROR_TEST_DATA_SET_X,
                                                       test_data.ZERO_ERROR_TEST_DATA_SET_Y,
                                                       test_data.ZERO_ERROR_TEST_DATA_SET_THETA)
        self.assertEqual(True, num_py.array_equal(j_history, test_data.ZERO_ITERATIONS_ZERO_ALPHA_EXPECTED))
        self.assertEqual(True, num_py.array_equal(theta, test_data.ZERO_ERROR_TEST_DATA_SET_THETA))

    def test_gradient_descent_two_iterations_with_small_alpha(self):
        alpha = 0.2
        lambda_param = 0.001
        no_of_iterations = 2
        lr_object = lr.LinearRegression(alpha, no_of_iterations, lambda_param)
        [j_history, theta] = lr_object.gradient_decent(test_data.ZERO_ERROR_TEST_DATA_SET_X,
                                                       test_data.ZERO_ERROR_TEST_DATA_SET_Y,
                                                       test_data.ZERO_ERROR_TEST_DATA_SET_THETA)
        self.assertEqual(True, num_py.array_equal(theta, test_data.TWO_ITERATIONS_SMALL_ALPHA_EXPECTED))

    def test_gradient_descent_fifty_iterations_with_big_alpha(self):
        alpha = 0.2
        lambda_param = 0.001
        no_of_iterations = 50
        lr_object = lr.LinearRegression(alpha, no_of_iterations, lambda_param)
        [j_history, theta] = lr_object.gradient_decent(test_data.ZERO_ERROR_TEST_DATA_SET_X,
                                                       test_data.ZERO_ERROR_TEST_DATA_SET_Y,
                                                       test_data.ZERO_ERROR_TEST_DATA_SET_THETA)
        self.assertEqual(True, num_py.array_equal(theta, test_data.FIFTY_ITERATIONS_SMALL_ALPHA_EXPECTED))

    def test_gradient_descent_two_iterations_with_big_alpha(self):
        alpha = 1
        lambda_param = 0.001
        no_of_iterations = 2
        lr_object = lr.LinearRegression(alpha, no_of_iterations, lambda_param)
        [j_history, theta] = lr_object.gradient_decent(test_data.ZERO_ERROR_TEST_DATA_SET_X,
                                                       test_data.ZERO_ERROR_TEST_DATA_SET_Y,
                                                       test_data.ZERO_ERROR_TEST_DATA_SET_THETA)
        self.assertEqual(True, num_py.array_equal(theta, test_data.TWO_ITERATIONS_BIG_ALPHA_EXPECTED))

