import unittest
import machine_learning_tests.test_data.cnn_test_data as constants
import machine_learning.conv_nn.cnn_kwargs_builder as args_builder


class KwargsBuilderTests(unittest.TestCase):

    def cnn_parameters_parsing_test(self):
        [layer_tuples, layer_properties] = \
            args_builder.ConvNetArgsBuilder(constants.CONV_NN_PARAMETERS).cnn_parameters_parser()
        self.assertEqual(len(layer_tuples), 9)
        self.assertEqual(len(layer_properties), 21)
