import machine_learning.nueral_network.simple_neural_network as simple_ann
import machine_learning.common_utils.common_constants as constants


class NeuralNetworkFactory:
    def __init__(self):
        pass

    @staticmethod
    def build_ann(self, params, ann_type):
        if ann_type and ann_type == constants.SIMPLE_ANN:
            return simple_ann.SimpleNeuralNetwork(params)
        return