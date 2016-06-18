import unittest
import sklearn.linear_model as lm
import machine_learning_tests.test_data.lin_reg_data as constants
import machine_learning.linear_regression.lin_reg_builder as lr_builder


class LinearRegressionBuilderTests(unittest.TestCase):

    def linear_reg_builder_tests(self):
        linear_reg = lr_builder.LinearRegressionBuilder(constants.LINEAR_REGRESSION_ARGS).build()
        self.assertEqual((lm.LinearRegression), type(linear_reg))
