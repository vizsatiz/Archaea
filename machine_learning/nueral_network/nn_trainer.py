import pybrain.supervised.trainers as trainers
import pybrain.utilities as utilities


class NeuralNetworkTrainer:

    def __init__(self, parameters):
        self.network_module = parameters['network']
        self.dataset = parameters['dataset']
        self.momentum = parameters['momentum']
        self.learning_rate = parameters['learningrate']
        self.verbose = parameters['verbose']
        self.weight_decay = parameters['weightdecay']
        # This trainer object is of PyBrain
        self.trainer = self.__initiate_and_get_trainer()

    def __initiate_and_get_trainer(self):
        """
        Building the back propagation trainer

        @dataset : The data on which the network is going to train from
        @momentum :
        @learning rate : The rate at which the error is going to be reduced (J(theta) getting converged)
        @verbose :
        @weightdecay :
        :return:
        """
        # TODO Find the use of remaining parameters and the reaseon for there existence and effect on the system
        return trainers.BackpropTrainer(self.network_module,
                                 dataset=self.dataset,
                                 momentum=self.momentum,
                                 learningrate=self.learning_rate,
                                 verbose=self.verbose,
                                 weightdecay=self.weight_decay)

    def train(self, epochs_count):
        """
        Using the back propagation trainer to train the network to a given epoch

        :param epochs_count: The number of iterations to be run for convergence.
        :return:
        """
        j_history = []
        for count in range(epochs_count):
            j_history.append(self.trainer.train())
        return j_history

    def percentage_error_on_dataset(self, datasets):
        """
        Function returns the percentage error on neural network with respect to given data

        :param datasets:
        :return:
        """
        return utilities.percentError(self.trainer.testOnClassData(dataset=datasets)
                    ,datasets['class'])
