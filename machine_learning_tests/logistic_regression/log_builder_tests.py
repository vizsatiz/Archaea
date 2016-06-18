import unittest
import sklearn.linear_model as lm
import machine_learning_tests.test_data.log_reg_data as constants
import machine_learning.logistic_regression.log_reg_builder as builder


class LogisticRegressionBuilder(unittest.TestCase):

    def logistic_regression_tests(self):
        log_reg = builder.LogisticRegressionBuilder(constants.LOGISTIC_REGRESSION_PARAMS).build()
        self.assertEqual((lm.LogisticRegression), type(log_reg))
