import unittest
import machine_learning.nueral_network.nn_trainer as bp_trainer
import machine_learning.nueral_network.nn_trainer as trainer_lib
import machine_learning.nueral_network.network_builder as builder


class TestNeuralNetworkTrainerFactory(unittest.TestCase):

    def test_get_ann_trainer(self):
        dimension = [128, 64, 32]
        fnn = builder.NeuralNetworkBuilder(dimensions=dimension).build()
        parameters = {'network': fnn, 'dataset': '', 'momentum': 0.1, 'learningrate': 0.01, 'verbose': True,
                      'weightdecay': 0.01}
        trainer = trainer_lib.NeuralNetworkTrainer(parameters)
        self.assertEqual(True, isinstance(trainer, bp_trainer.NeuralNetworkTrainer))

