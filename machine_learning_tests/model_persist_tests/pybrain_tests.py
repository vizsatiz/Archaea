import unittest
from numpy import ravel
from sklearn import datasets
from pybrain.datasets import ClassificationDataSet
import machine_learning.nueral_network.nn_trainer as trainer
import machine_learning.nueral_network.network_builder as builder
import machine_learning.model_persistance.pybrain_model as persistor


class PyBrainNetworkPersistenceHelperTests(unittest.TestCase):

    def pybrain_lr_model_persistence_tests(self):

        olivetti = datasets.fetch_olivetti_faces()
        X, y = olivetti.data, olivetti.target
        ds = ClassificationDataSet(4096, 1, nb_classes=40)
        for k in xrange(len(X)):
            ds.addSample(ravel(X[k]), y[k])
        tstdata, trndata = ds.splitWithProportion(0.25)
        trndata._convertToOneOfMany()
        tstdata._convertToOneOfMany()
        dimension = [trndata.indim, 64, trndata.outdim]
        fnn = builder.NeuralNetworkBuilder(dimensions=dimension).build()
        parameters = {'network': fnn, 'dataset': trndata, 'momentum': 0.1,
                      'learningrate': 0.01, 'verbose': True, 'weightdecay': 0.01}
        bp_trainer = trainer.NeuralNetworkTrainer(parameters)
        bp_trainer.train(2)
        efficiency_before_saving = bp_trainer.percentage_error_on_dataset(tstdata)
        state_object = persistor.PyBrainNetworkPersistenceHelper.get_model_state(bp_trainer.network_module)
        fnn_new  = persistor.PyBrainNetworkPersistenceHelper.initialize_model_with_state(state_object)
        parameters_new = {'network': fnn_new, 'dataset': trndata, 'momentum': 0.1,
                      'learningrate': 0.01, 'verbose': True, 'weightdecay': 0.01}
        bp_trainer_new = trainer.NeuralNetworkTrainer(parameters_new)
        efficiency_after_saving = bp_trainer_new.percentage_error_on_dataset(tstdata)
        self.assertEqual(efficiency_before_saving, efficiency_after_saving)