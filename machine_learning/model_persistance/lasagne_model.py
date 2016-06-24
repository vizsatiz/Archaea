class LasagneModelPersistenceHelper:

    def __init__(self):
        pass

    @staticmethod
    def get_model_state(model_object):
        """
        This methods dumps the current state of model and  returns it

        :param model_object:
        :return:
        """
        return model_object.get_all_params_values()

    @staticmethod
    def initialize_model_with_state(model_object, state):
        """
        This method re-initiates the model with the state passed.

        :param state:
        :return:
        """
        return model_object.load_weights_from(state)