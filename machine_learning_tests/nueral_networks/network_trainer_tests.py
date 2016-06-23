import unittest
import machine_learning.nueral_network.nn_trainer as bp_trainer
import machine_learning.nueral_network.nn_trainer as trainer_lib


class TestNeuralNetworkTrainerFactory(unittest.TestCase):

    def test_get_ann_trainer(self):
        parameters = {'network': '', 'dataset': '', 'momentum': 0.1, 'learningrate': 0.01, 'verbose': True,
                      'weightdecay': 0.01}
        trainer = trainer_lib.NeuralNetworkTrainer(parameters)
        self.assertEqual(True, isinstance(trainer, bp_trainer.NeuralNetworkTrainer))

