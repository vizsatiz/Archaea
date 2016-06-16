import machine_learning.common_utils.common_constants as constants
import machine_learning.common_utils.error_messages as errors
from pybrain.tools.shortcuts import buildNetwork as pybrain_builder
from pybrain.structure.modules import SoftmaxLayer


class NetworkBuilderFactory:

    def __init__(self, dimensions, ann_type):
        self.dimensions = dimensions
        self.ann_type = ann_type

    def build_ann(self):
        """
        Builds and returns a network with the passed architecture

        :return:
        """
        if self.ann_type and self.ann_type == constants.SIMPLE_ANN:
            return SimpleNetworkBuilder(self.dimensions).build_network()
        return None


class SimpleNetworkBuilder:

    def __init__(self, network_dimensions):
        self.network_dimensions = network_dimensions

    def validate_dimensions(self):
        """
        Validating the network dimensions

        1. 0 < length of dimensions <= 10
        2. each dimension > 0

        :return:
        """
        if len(self.network_dimensions) < 2 or len(self.network_dimensions) >= constants.NEURAL_NETWORK_MAX_LAYER_COUNT:
            raise ValueError(errors.NEURAL_NETWORK_DIMENSIONS_MISMATCH)
        for number_of_nodes in self.network_dimensions:
            if number_of_nodes < 0:
                raise ValueError(errors.NETWORK_DIMENSIONS_NEGATIVE)

    def build_network(self):
        """
        Builds and returns the network of required size

        :return:
        """
        self.validate_dimensions()
        return pybrain_builder(*self.network_dimensions, outclass=SoftmaxLayer)
