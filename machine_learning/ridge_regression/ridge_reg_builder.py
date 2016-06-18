import sklearn.linear_model as lm


class RidgeRegressionBuilder:

    def __init__(self):
        """
            Initialize the logistic regression

            :param network_architecture: The architecture of the network
            """
        pass

    def build(self):
        """
        Linear regression object builder

        :return:
        """
        return lm.RidgeCV()
