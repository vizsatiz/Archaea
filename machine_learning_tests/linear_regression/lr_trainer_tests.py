import unittest
import numpy as num_py
import machine_learning_tests.test_data.lin_reg_data as constants
import machine_learning.linear_regression.lin_reg_trainer as trainer
import machine_learning.linear_regression.lin_reg_builder as lr_builder


class TestLinearRegressionTrainerTest(unittest.TestCase):

    def test_train_lr_function(self):
        linear_reg = lr_builder.LinearRegressionBuilder(constants.LINEAR_REGRESSION_ARGS).build()
        f = lambda x: num_py.exp(3 * x)
        x_tr = num_py.linspace(0., 2, 200)
        y_tr = f(x_tr)
        x = num_py.array([0, .1, .2, .5, .8, .9, 1])
        y = f(x) + num_py.random.randn(len(x))
        lr_trainer = trainer.LinearRegressionTrainer(linear_reg, 3)
        lr_trainer.train(x,y)
        self.assertEqual(len(lr_trainer.predict(x)), 7)