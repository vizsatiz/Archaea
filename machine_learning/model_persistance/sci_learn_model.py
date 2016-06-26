import pickle


class SciLearnModelPersistenceHelper:

    def __init__(self):
        pass

    @staticmethod
    def get_model_state(model_object):
        """
        This methods dumps the current state of model and  returns it

        :param model_object:
        :return:
        """
        return pickle.dumps(model_object)

    @staticmethod
    def initialize_model_with_state(state):
        """
        This method re-initiates the model with the state passed.

        :param state:
        :return:
        """
        return pickle.loads(state)