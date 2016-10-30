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
        degree = 3
        x = num_py.vander(x, degree + 1)
        lr_trainer = trainer.LinearRegressionTrainer(linear_reg)
        lr_trainer.train(x, y)
        self.assertEqual(len(lr_trainer.predict(x)), 7)

    def error_and_variance_tests(self):
        linear_reg = lr_builder.LinearRegressionBuilder(constants.LINEAR_REGRESSION_ARGS).build()
        f = lambda x: num_py.exp(3 * x)
        x = num_py.array([0, .1, .2, .5, .8, .9, 1])
        y = f(x) + num_py.random.randn(len(x))
        degree = 5
        x = num_py.vander(x, degree + 1)
        lr_trainer = trainer.LinearRegressionTrainer(linear_reg)
        lr_trainer.train(x, y)
        [mean_error, variance] = lr_trainer.error_and_variance(x, y)
        self.assertLess(mean_error, 2)
        self.assertLess(variance, 1.5)

    def get_coefficient_tests(self):
        linear_reg = lr_builder.LinearRegressionBuilder(constants.LINEAR_REGRESSION_ARGS).build()
        f = lambda x: num_py.exp(3 * x)
        x = num_py.array([0, .1, .2, .5, .8, .9, 1])
        y = f(x) + num_py.random.randn(len(x))
        degree = 5
        x = num_py.vander(x, degree + 1)
        lr_trainer = trainer.LinearRegressionTrainer(linear_reg)
        lr_trainer.train(x, y)
        coefficients = lr_trainer.regression_coefficients()
        self.assertEqual(len(coefficients), 6)

    def test_train_lr_function_multidimensional_input(self):
        linear_reg = lr_builder.LinearRegressionBuilder(constants.LINEAR_REGRESSION_ARGS).build()
        x = num_py.array([[0, .1, .2], [.5, .8, .9], [1, 1.2, 1.5]])
        y = [4.17197761, 30.38459717, 146.70090266]
        lr_trainer = trainer.LinearRegressionTrainer(linear_reg)
        lr_trainer.train(x, y)
        coefficients = lr_trainer.regression_coefficients()
        self.assertEqual(len(coefficients), 3)
        self.assertEqual(len(lr_trainer.predict(x)), 3)
        self.assertEqual(lr_trainer.predict(x)[0], 4.1719776100001837)
