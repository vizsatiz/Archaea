import pickle
import unittest
import numpy as num_py
import machine_learning_tests.test_data.lin_reg_data as constants
import machine_learning.linear_regression.lin_reg_trainer as trainer
import machine_learning.linear_regression.lin_reg_builder as lr_builder
import machine_learning.model_persistance.sci_learn_model as persistor


class SciLearnModelPersistenceHelperTests(unittest.TestCase):

    def sci_learn_lr_model_persistence_tests(self):

        linear_reg = lr_builder.LinearRegressionBuilder(constants.LINEAR_REGRESSION_ARGS).build()
        f = lambda x: num_py.exp(3 * x)
        x = num_py.array([0, .1, .2, .5, .8, .9, 1])
        y = f(x)
        lr_trainer = trainer.LinearRegressionTrainer(linear_reg, 3)
        lr_trainer.train(x, y)
        x_predict = num_py.array([.9, 1])
        prediction_before_saving = lr_trainer.predict(x_predict)
        object = persistor.SciLearnModelPersistenceHelper.get_model_state(lr_trainer.lr_object)
        lr = persistor.SciLearnModelPersistenceHelper.initialize_model_with_state(object)
        lr_trainer_new = trainer.LinearRegressionTrainer(lr, 3)
        prediction_after_saving  = lr_trainer_new.predict(x_predict)
        self.assertEqual(prediction_before_saving.item(0), prediction_after_saving.item(0))

    def sci_learn_lr_model_file_to_object_conversion_test(self):
        linear_reg = lr_builder.LinearRegressionBuilder(constants.LINEAR_REGRESSION_ARGS).build()
        f = lambda x: num_py.exp(3 * x)
        x = num_py.array([0, .1, .2, .5, .8, .9, 1])
        y = f(x)
        lr_trainer = trainer.LinearRegressionTrainer(linear_reg, 3)
        lr_trainer.train(x, y)
        x_predict = num_py.array([.9, 1])
        prediction_before_saving = lr_trainer.predict(x_predict)
        object = persistor.SciLearnModelPersistenceHelper.get_model_state(lr_trainer.lr_object)
        model_state = pickle.dumps(object)
        object_reloaded = pickle.loads(model_state)
        lr = persistor.SciLearnModelPersistenceHelper.initialize_model_with_state(object_reloaded)
        lr_trainer_new = trainer.LinearRegressionTrainer(lr, 3)
        prediction_after_saving = lr_trainer_new.predict(x_predict)
        self.assertEqual(prediction_before_saving.item(0), prediction_after_saving.item(0))


