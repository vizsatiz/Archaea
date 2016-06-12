import unittest
import numpy as num_py
import tests.test_data.com_util_data as test_data
import machine_learning.common_utils.lambda_regularization as lambda_reg
import machine_learning.common_utils.common_helper as common_helper
import machine_learning.common_utils.error_messages as errors


class TestCommonUtils(unittest.TestCase):

    def lambda_cost_regularization_test(self):
        regularization_value = lambda_reg.Regularization(test_data.REGULARIZATION_TEST_DATA_FOR_LAMBDA). \
            cost_regularization(test_data.REGULARIZATION_TEST_DATA_FOR_THETA)
        self.assertEqual(regularization_value, test_data.REGULARIZATION_TEST_DATA_FOR_COST_EXPECTED)

    def lambda_gradient_regularization_test(self):
        regularization_value = lambda_reg.Regularization(test_data.REGULARIZATION_TEST_DATA_FOR_LAMBDA). \
            gradient_regularization(test_data.REGULARIZATION_TEST_DATA_FOR_THETA)
        self.assertEqual(True, num_py.array_equal(regularization_value,
                                                  test_data.REGULARIZATION_TEST_DATA_FOR_GRADIENT_EXPECTED))

    def validate_matrix_multiplication_dimensions_negative_test(self):
        matrix_1 = num_py.matrix('1 2')
        try:
            common_helper.validate_matrix_multiplication_dimensions(matrix_1, matrix_1)
            self.fail()
        except ValueError as err:
            self.assertEqual(err.message, errors.MATRIX_DIMENSION_MISMATCH_ERROR)

    def validate_matrix_multiplication_dimensions_positive_test(self):
        matrix_1 = num_py.matrix('1 2')
        matrix_2 = num_py.matrix('1;2')
        try:
            common_helper.validate_matrix_multiplication_dimensions(matrix_1, matrix_2)
        except ValueError as err:
            self.fail()


if __name__ == '__main__':
    unittest.main()
