import unittest
from numpy import ravel
from sklearn import datasets
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.xml.networkwriter import NetworkWriter
import machine_learning.nueral_network.simple_neural_network as snn
import machine_learning.model.NetworkArchitecture as netArch
import machine_learning.nueral_network.nn_trainer_factory as trainFact
import machine_learning.common_utils.common_constants as constants


class NeuralNetworkIntegrationTests(unittest.TestCase):

    def test_neural_network_training(self):
        olivetti = datasets.fetch_olivetti_faces()
        X, y = olivetti.data, olivetti.target
        ds = ClassificationDataSet(4096, 1, nb_classes=40)
        for k in xrange(len(X)):
            ds.addSample(ravel(X[k]), y[k])
        tstdata, trndata = ds.splitWithProportion(0.25)
        trndata._convertToOneOfMany()
        tstdata._convertToOneOfMany()
        dimension = [trndata.indim, 64, trndata.outdim]
        fnn = snn.SimpleNeuralNetwork(netArch.NetworkArchitecture(dimension)).get_simple_neural_network()
        parameters = {'network': fnn, 'dataset': trndata, 'momentum': 0.1, 'learningrate': 0.01, 'verbose': True,
                      'weightdecay': 0.01}
        trainer = trainFact.NetworkTrainer(parameters).get_ann_trainer(constants.BACK_PROP_TRAINER)
        errors = trainer.train(2)
        efficiency = trainer.percentage_error_on_dataset(tstdata)
        self.assertGreaterEqual(efficiency, 90.0)
