import machine_learning.nueral_network.network_builder as network_builder
import machine_learning.common_utils.common_constants as constants


class SimpleNeuralNetwork:
    def __init__(self, network_architecture):
        self.network_architecture = network_architecture

    def __network_builder(self):
        """
        Builds the NN given the network architecture

        :return:
        """
        return network_builder.NetworkBuilderFactory(self.network_architecture, constants.SIMPLE_ANN).build_ann()

    def get_simple_neural_network(self):
        """
        Returns the neural network

        :return:
        """
        return self.__network_builder()
