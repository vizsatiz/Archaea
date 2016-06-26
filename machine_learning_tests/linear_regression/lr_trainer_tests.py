import unittest
import numpy as num_py
import machine_learning_tests.test_data.lin_reg_data as constants
import machine_learning.linear_regression.lin_reg_trainer as trainer
import machine_learning.linear_regression.lin_reg_builder as lr_builder


class TestLinearRegressionTrainerTest(unittest.TestCase):

    def test_train_lr_function(self):
        linear_reg = lr_builder.LinearRegressionBuilder(constants.LINEAR_REGRESSION_ARGS).build()
        f = lambda x: num_py.exp(3 * x)
        x = num_py.array([0, .1, .2, .5, .8, .9, 1])
        y = f(x) + num_py.random.randn(len(x))
        lr_trainer = trainer.LinearRegressionTrainer(linear_reg, 3)
        lr_trainer.train(x, y)
        self.assertEqual(len(lr_trainer.predict(x)), 7)

    def error_and_variance_tests(self):
        linear_reg = lr_builder.LinearRegressionBuilder(constants.LINEAR_REGRESSION_ARGS).build()
        f = lambda x: num_py.exp(3 * x)
        x = num_py.array([0, .1, .2, .5, .8, .9, 1])
        y = f(x) + num_py.random.randn(len(x))
        lr_trainer = trainer.LinearRegressionTrainer(linear_reg, 5)
        lr_trainer.train(x, y)
        [mean_error, variance] = lr_trainer.error_and_variance(x, y)
        self.assertLess(mean_error, 2)
        self.assertLess(variance, 1.5)

    def get_coefficient_tests(self):
        linear_reg = lr_builder.LinearRegressionBuilder(constants.LINEAR_REGRESSION_ARGS).build()
        f = lambda x: num_py.exp(3 * x)
        x = num_py.array([0, .1, .2, .5, .8, .9, 1])
        y = f(x) + num_py.random.randn(len(x))
        lr_trainer = trainer.LinearRegressionTrainer(linear_reg, 5)
        lr_trainer.train(x, y)
        coefficients = lr_trainer.regression_coefficients()
        self.assertEqual(len(coefficients), 6)

