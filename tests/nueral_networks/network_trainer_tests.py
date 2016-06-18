import machine_learning.nueral_network.back_propagation_trainer as bp_trainer
import machine_learning.nueral_network.nn_trainer_factory as trainer_lib
import machine_learning.common_utils.common_constants as constants
import unittest


class TestNeuralNetworkTrainerFactory(unittest.TestCase):

    def test_get_ann_trainer(self):
        parameters = {'network': '', 'dataset': '', 'momentum': 0.1, 'learningrate': 0.01, 'verbose': True,
                      'weightdecay': 0.01}
        trainer = trainer_lib.NetworkTrainer(parameters).get_ann_trainer(constants.BACK_PROP_TRAINER)
        self.assertEqual(True, isinstance(trainer, bp_trainer.BackPropagationTrainer))

