import machine_learning.common_utils.common_constants as constants
import machine_learning.common_utils.error_messages as errors
from pybrain.tools.shortcuts import buildNetwork as network_builder
from pybrain.structure.modules import SoftmaxLayer


class NeuralNetworkBuilder:

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def validate_dimensions(self):
        """
        Validating the network dimensions

        1. 0 < length of dimensions <= 10
        2. each dimension > 0

        :return:
        """
        if len(self.dimensions) < 2 or len(self.dimensions) >= constants.NEURAL_NETWORK_MAX_LAYER_COUNT:
            raise ValueError(errors.NEURAL_NETWORK_DIMENSIONS_MISMATCH)
        for number_of_nodes in self.dimensions:
            if number_of_nodes < 0:
                raise ValueError(errors.NETWORK_DIMENSIONS_NEGATIVE)

    def build(self):
        self.validate_dimensions()
        return network_builder(*self.dimensions, outclass=SoftmaxLayer)