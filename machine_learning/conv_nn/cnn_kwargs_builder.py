import conv_constants as constants
import layer_factory as layer_factory
import lasagne.updates as updates


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
        # Constructing kwargs with layer properties and layer name tuples
        for cnn_layer in layer_objects:
            layer_name = cnn_layer[constants.LAYER_NAME]
            layer_tuple = (layer_name, layer_factory.LayerFactory(layer_name).layer_object_mapper())
            layers_tuple_list.append(layer_tuple)
            if cnn_layer[constants.LAYER_TYPE] == constants.INPUT_LAYER:
                layer_properties[layer_name + '_shape'] = cnn_layer[constants.SHAPE]
            elif cnn_layer[constants.LAYER_TYPE] == constants.CONVOLUTIONAL_2D_LAYER:
                layer_properties[layer_name + '_num_filters'] = cnn_layer[constants.NUMBER_OF_FILTERS]
                layer_properties[layer_name + '_filter_size'] = cnn_layer[constants.FILTER_SIZE]
                layer_properties[layer_name + '_nonlinearity'] = cnn_layer[constants.LAYER_NONLINEARITY]
                layer_properties[layer_name + '_W'] = cnn_layer[constants.CONV_WINDOW]
            elif cnn_layer[constants.LAYER_TYPE] == constants.MAX_POOL_2D_LAYER:
                layer_properties[layer_name + '_pool_size'] = cnn_layer[constants.POOL_SIZE]
            elif cnn_layer[constants.LAYER_TYPE] == constants.DROPOUT_LAYER:
                layer_properties[layer_name + '_p'] = cnn_layer[constants.DROPOUT_PIVOT]
            elif cnn_layer[constants.LAYER_TYPE] == constants.DENSE_LAYER:
                layer_properties[layer_name + '_num_units'] = cnn_layer[constants.NUM_UNITS]
                layer_properties[layer_name + '_nonlinearity'] = cnn_layer[constants.LAYER_NONLINEARITY]
            else:
                print 'Layer not found : ' + cnn_layer[constants.LAYER_TYPE]
        # Adding other specifications for the CNN
        layer_properties['update'] = self.update_mapper(self.cnn_architecture['update'])
        layer_properties['update_learning_rate'] = self.cnn_architecture['update_learning_rate']
        layer_properties['update_momentum'] = self.cnn_architecture['update_momentum']
        layer_properties['max_epochs'] = self.cnn_architecture['max_epochs']
        layer_properties['verbose'] = self.cnn_architecture['verbose']
        return layers_tuple_list, layer_properties

    @staticmethod
    def update_mapper(update_name):
        """
        Update function can be mapped to corresponding function here

        :param update_name:
        :return:
        """
        layer_mapper = {
            'nesterov_momentum': updates.nesterov_momentum, #Stochastic Gradient Descent (SGD) updates with Nesterov momentum
            'sgd': updates.sgd, # Stochastic Gradient Descent (SGD)
            'momentum': updates.momentum, #Stochastic Gradient Descent (SGD) updates with momentum
            'adam' : updates.adam, #adams update
            'adamax' : updates.adamax
        }
        return layer_mapper[update_name]