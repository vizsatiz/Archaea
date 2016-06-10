from lasagne import layers
import conv_constants as constants


class LayerFactory:
    def __init__(self, layer_name):
        """
        Class used for creating a layer architecture for convolutional NN
        :param layer_architecture: This should be a list of strings.
        """
        self.layer_name = layer_name
        # Maps different layer names to its objects
        self.layer_mapper = {
            constants.INPUT_LAYER: layers.InputLayer,
            constants.CONVOLUTIONAL_2D_LAYER: layers.Conv2DLayer,
            constants.MAX_POOL_2D_LAYER: layers.MaxPool2DLayer,
            constants.DENSE_LAYER: layers.DenseLayer,
            constants.DROPOUT_LAYER: layers.DropoutLayer
        }

    def layer_object_mapper(self):
        """
        Returns the layer object by taking layer name as string

        :param layer_name: layer name as string
        :return:
        """
        return self.layer_mapper[self.layer_name]

