import machine_learning.common_utils.common_constants as constants
import machine_learning.nueral_network.back_propagation_trainer as bp_trainer


class NetworkTrainer:
    def __init__(self, parameters):
        self.parameters = parameters

    def get_ann_trainer(self, trainer_type):
        """
        This function returns the back propagation trainer object

        :param trainer_type:
        :return:
        """
        if trainer_type and trainer_type == constants.BACK_PROP_TRAINER:
            return bp_trainer.BackPropagationTrainer(parameters=self.parameters)
        return None
