import sklearn.linear_model as lm


class LinearRegressionBuilder:

    def __init__(self, network_architecture):
        """
            Initialize the logistic regression

            :param network_architecture: The architecture of the network
            """
        self.network_architecture = network_architecture

    def build(self):
        """
        Linear regression object builder

        :return:
        """
        return lm.LinearRegression(fit_intercept=self.network_architecture['fit_intercept'],
                                   normalize=self.network_architecture['normalize'],
                                   copy_X=self.network_architecture['copy_X'],
                                   n_jobs=self.network_architecture['n_jobs'])
