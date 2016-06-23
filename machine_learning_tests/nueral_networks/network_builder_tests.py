import machine_learning.nueral_network.network_builder as network_builder
import machine_learning.common_utils.error_messages as errors
import pybrain as py_brain
import unittest


class TestNeuralNetworkBuilder(unittest.TestCase):

    def test_network_builder_factory(self):
        dimensions = [128, 64, 10]
        ann = network_builder.NeuralNetworkBuilder(dimensions).build()
        self.assertEqual(True, isinstance(ann, py_brain.FeedForwardNetwork))

    def test_simple_network_builder(self):
        dimensions = [128, 64, 10]
        ann = network_builder.NeuralNetworkBuilder(dimensions).build()
        self.assertEqual(True, isinstance(ann, py_brain.FeedForwardNetwork))

    def test_dimensions_validator_negative(self):
        dimensions = [2]
        try:
            ann = network_builder.NeuralNetworkBuilder(dimensions).build()
            self.fail('Network Dimension Validator Failure')
        except ValueError as err:
            self.assertEqual(err.message, errors.NEURAL_NETWORK_DIMENSIONS_MISMATCH)

    def test_dimensions_validator_positive(self):
        dimensions = [2, 6, 7]
        try:
            ann = network_builder.NeuralNetworkBuilder(dimensions).build()
        except ValueError as err:
            self.fail('Network Dimension Validator Failure')


if __name__ == '__main__':
    unittest.main()
