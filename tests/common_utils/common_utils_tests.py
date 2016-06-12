import unittest
import numpy as num_py
import cu_test_data as test_data
import machine_learning.common_utils.lambda_regularization as lambda_reg


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


if __name__ == '__main__':
    unittest.main()
