import machine_learning.common_utils.common_constants as constants
from pybrain.tools.shortcuts import buildNetwork as pyBrain_builder
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
        pass

    def build_network(self):
        """
        Builds and returns the network of required size

        :return:
        """
        self.validate_dimensions()
        print self.network_dimensions
        print type(self.network_dimensions)
        return pyBrain_builder(*self.network_dimensions, outclass=SoftmaxLayer)
