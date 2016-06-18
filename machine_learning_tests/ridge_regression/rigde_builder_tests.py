import unittest
import sklearn.linear_model as lm
import machine_learning_tests.test_data.lin_reg_data as constants
import machine_learning.ridge_regression.ridge_reg_builder as ridge_builder


class LinearRegressionBuilderTests(unittest.TestCase):

    def linear_reg_builder_tests(self):
        linear_reg = ridge_builder.RidgeRegressionBuilder().build()
        self.assertEqual((lm.RidgeCV), type(linear_reg))
