from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader


class PyBrainNetworkPersistenceHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_model_state(model_object):
        """
        Method that gets the current state of the network

        :param model_object:
        :return:
        """
        return NetworkWriterHelper.get_network_state(model_object, 'filename')

    @staticmethod
    def initialize_model_with_state(dom):
        """
        Method re-initiates the network from dom

        :param dom:
        :return:
        """
        file = open('some.xml', 'w')
        file.write(dom)
        file.close()
        return NetworkReader.readFrom('some.xml')


class NetworkWriterHelper(NetworkWriter):
    @staticmethod
    def get_network_state(net, filename):
        """
        Returns the Network State as XML

        :param net:
        :param filename:
        :return:
        """
        w = NetworkWriter(filename, newfile=True)
        netroot = w.newRootNode('Network')
        w.writeNetwork(net, netroot)
        return w.dom.toprettyxml()
