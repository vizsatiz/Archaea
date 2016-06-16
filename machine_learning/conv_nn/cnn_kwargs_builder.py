import lasagne
from lasagne import layers
import lasagne.updates as updates
import conv_constants as constants
import machine_learning.common_utils.error_messages as error

class ConvNetArgsBuilder:

    def __init__(self, cnn_architecture):
        self.cnn_architecture = cnn_architecture

    def cnn_parameters_parser(self):
        """
        This method converts key value pair of cnn_params to lists which can be used to
        train the network

        :return:
        """
        layers_tuple_list = []
        layer_properties = {}
        layer_objects = self.cnn_architecture['layers']
        if layer_objects:
        # Constructing kwargs with layer properties and layer name tuples
            for cnn_layer in layer_objects:
                layer_name = cnn_layer[constants.LAYER_NAME]
                layer_tuple = (layer_name, self.layer_object_mapper(cnn_layer[constants.LAYER_TYPE]))
                layers_tuple_list.append(layer_tuple)
                if cnn_layer[constants.LAYER_TYPE] == constants.INPUT_LAYER:
                    layer_properties[layer_name + '_shape'] = cnn_layer[constants.SHAPE]
                elif cnn_layer[constants.LAYER_TYPE] == constants.CONVOLUTIONAL_2D_LAYER:
                    layer_properties[layer_name + '_num_filters'] = cnn_layer[constants.NUMBER_OF_FILTERS]
                    layer_properties[layer_name + '_filter_size'] = cnn_layer[constants.FILTER_SIZE]
                    if cnn_layer[constants.LAYER_NONLINEARITY]:
                        layer_properties[layer_name + '_nonlinearity'] = self.non_linearity_mapper\
                        (cnn_layer[constants.LAYER_NONLINEARITY])
                    if cnn_layer[constants.CONV_WINDOW]:
                        layer_properties[layer_name + '_W'] = self.conv_window_mapper(cnn_layer[constants.CONV_WINDOW])
                elif cnn_layer[constants.LAYER_TYPE] == constants.MAX_POOL_2D_LAYER:
                    layer_properties[layer_name + '_pool_size'] = cnn_layer[constants.POOL_SIZE]
                elif cnn_layer[constants.LAYER_TYPE] == constants.DROPOUT_LAYER:
                    layer_properties[layer_name + '_p'] = cnn_layer[constants.DROPOUT_PIVOT]
                elif cnn_layer[constants.LAYER_TYPE] == constants.DENSE_LAYER:
                    layer_properties[layer_name + '_num_units'] = cnn_layer[constants.NUM_UNITS]
                    if self.non_linearity_mapper:
                        layer_properties[layer_name + '_nonlinearity'] = self.non_linearity_mapper\
                            (cnn_layer[constants.LAYER_NONLINEARITY])
                else:
                    print 'Layer not found : ' + cnn_layer[constants.LAYER_TYPE]
            # Adding other specifications for the CNN
            layer_properties['update'] = self.update_mapper(self.cnn_architecture['update'])
            layer_properties['update_learning_rate'] = self.cnn_architecture['update_learning_rate']
            layer_properties['update_momentum'] = self.cnn_architecture['update_momentum']
            layer_properties['max_epochs'] = self.cnn_architecture['max_epochs']
            layer_properties['verbose'] = self.cnn_architecture['verbose']
            return layers_tuple_list, layer_properties
        else:
            raise ValueError(error.NEURAL_NETWORK_DIMENSIONS_MISMATCH)

    @staticmethod
    def update_mapper(update_name):
        """
        Update function can be mapped to corresponding function here

        :param update_name:
        :return:
        """
        layer_map = {
            'nesterov_momentum': updates.nesterov_momentum, #Stochastic Gradient Descent (SGD) updates with Nesterov momentum
            'sgd': updates.sgd, # Stochastic Gradient Descent (SGD)
            'momentum': updates.momentum, #Stochastic Gradient Descent (SGD) updates with momentum
            'adam': updates.adam, #adams update
            'adamax': updates.adamax
        }
        return layer_map[update_name]

    @staticmethod
    def non_linearity_mapper(non_linearity_name):
        """
        Non linearity function mapped to corresponding function object

        :param non_linearity_name:
        :return:
        """
        non_linearity_map = {
            'softmax': lasagne.nonlinearities.softmax,
            'rectify': lasagne.nonlinearities.rectify
        }
        return non_linearity_map[non_linearity_name]

    @staticmethod
    def conv_window_mapper(window_name):
        """
        Conv Window mapped to corresponding window object

        :param window_name:
        :return:
        """
        window_map = {
            'glorotuniform': lasagne.init.GlorotUniform()
        }
        return window_map[window_name]

    @staticmethod
    def layer_object_mapper(layer_name):
        """
        Returns the layer object by taking layer name as string

        :param layer_name: layer name as string
        :return:
        """
        layer_mapper = {
            constants.INPUT_LAYER: layers.InputLayer,
            constants.CONVOLUTIONAL_2D_LAYER: layers.Conv2DLayer,
            constants.MAX_POOL_2D_LAYER: layers.MaxPool2DLayer,
            constants.DENSE_LAYER: layers.DenseLayer,
            constants.DROPOUT_LAYER: layers.DropoutLayer
        }
        return layer_mapper[layer_name]