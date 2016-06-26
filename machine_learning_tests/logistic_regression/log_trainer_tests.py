import unittest
from sklearn import datasets
import machine_learning_tests.test_data.log_reg_data as constants
import machine_learning.logistic_regression.log_reg_trainer as trainer
import machine_learning.logistic_regression.log_reg_builder as log_builder


class LogisticRegressionTrainerTests(unittest.TestCase):

    def logistic_reg_trainer_tests(self):
        log_reg = log_builder.LogisticRegressionBuilder(constants.LOGISTIC_REGRESSION_PARAMS).build()
        lr_trainer = trainer.LogisticRegressionTrainer(log_reg)
        data_set = datasets.load_iris()
        lr_trainer.train(data_set.data, data_set.target)
        predicted = lr_trainer.predict(data_set.data)
        self.assertEqual(len(predicted), 150)

    def confusion_metrics_tests(self):
        log_reg = log_builder.LogisticRegressionBuilder(constants.LOGISTIC_REGRESSION_PARAMS).build()
        lr_trainer = trainer.LogisticRegressionTrainer(log_reg)
        data_set = datasets.load_iris()
        lr_trainer.train(data_set.data, data_set.target)
        [report, confusion_metrics] = lr_trainer.confusion_matrix(data_set.data, data_set.target)
        self.assertEqual(confusion_metrics[0][0], 50)
