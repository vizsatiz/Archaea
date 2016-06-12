import machine_learning.linear_regression.linear_regression as lr
import lr_test_data as test_data
import numpy as num_py
import unittest

alpha = 0
lambda_param = 0.001
no_of_iterations = 0
lr_object = lr.LinearRegression(alpha, no_of_iterations, lambda_param)


class TestLinearRegressionCostFunctions(unittest.TestCase):
    def test_cost_function_zero_error(self):
        cost_value = lr_object.cost(test_data.ZERO_ERROR_TEST_DATA_SET_X, test_data.ZERO_ERROR_TEST_DATA_SET_Y,
                                    test_data.ZERO_ERROR_TEST_DATA_SET_THETA)
        self.assertEqual(cost_value, test_data.ZERO_ERROR_TEST_DATA_SET_EXPECTED_OUTPUT)

    def test_cost_function_non_zero_error(self):
        cost_value = lr_object.cost(test_data.NON_ZERO_ERROR_TEST_DATA_SET_X, test_data.NON_ZERO_ERROR_TEST_DATA_SET_Y,
                                    test_data.NON_ZERO_ERROR_TEST_DATA_SET_THETA)
        self.assertEqual(cost_value, test_data.NON_ZERO_ERROR_TEST_DATA_SET_EXPECTED_OUTPUT)

    def test_regularized_cost_function_zero_error(self):
        cost_value = lr_object.regularized_cost(test_data.ZERO_ERROR_TEST_DATA_SET_X,
                                                test_data.ZERO_ERROR_TEST_DATA_SET_Y,
                                                test_data.ZERO_ERROR_TEST_DATA_SET_THETA)
        self.assertEqual(cost_value, test_data.ZERO_ERROR_TEST_DATA_SET_EXPECTED_OUTPUT_REGULARIZED)

    def test_regularized_cost_function_non_zero_error(self):
        cost_value = lr_object.regularized_cost(test_data.NON_ZERO_ERROR_TEST_DATA_SET_X,
                                                test_data.NON_ZERO_ERROR_TEST_DATA_SET_Y,
                                                test_data.NON_ZERO_ERROR_TEST_DATA_SET_THETA)
        self.assertEqual(cost_value, test_data.NON_ZERO_ERROR_TEST_DATA_SET_EXPECTED_OUTPUT_REGULARIZED)

    def test_h_of_theta(self):
        h_of_theta = lr_object.h_of_theta(test_data.NON_ZERO_ERROR_TEST_DATA_SET_X,
                                          test_data.NON_ZERO_ERROR_TEST_DATA_SET_THETA)
        self.assertEqual(True, num_py.array_equal(h_of_theta, test_data.H_OF_THETA))


if __name__ == '__main__':
    unittest.main()
