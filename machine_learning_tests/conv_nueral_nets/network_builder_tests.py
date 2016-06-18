import unittest
from nolearn.lasagne import NeuralNet
import machine_learning_tests.test_data.cnn_test_data as constants
import machine_learning.conv_nn.network_builder as builder


class ConvNeuralNetBuilderTests(unittest.TestCase):

    def network_builder_test(self):
        network = builder.ConvNetworkBuilder(constants.CONV_NN_PARAMETERS).build()
        self.assertEqual(isinstance(network, NeuralNet), True)
