import sklearn.linear_model as lm


class LogisticRegressionBuilder:

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
        return lm.LogisticRegression(penalty=self.network_architecture['penalty'],
                                     dual=self.network_architecture['dual'],
                                     tol=self.network_architecture['tol'],
                                     C=self.network_architecture['C'],
                                     fit_intercept=self.network_architecture['fit_intercept'],
                                     intercept_scaling=self.network_architecture['intercept_scaling'],
                                     class_weight=self.network_architecture['class_weight'],
                                     random_state=self.network_architecture['random_state'],
                                     solver=self.network_architecture['solver'],
                                     max_iter=self.network_architecture['max_iter'],
                                     multi_class=self.network_architecture['multi_class'],
                                     verbose=self.network_architecture['verbose'],
                                     warm_start=self.network_architecture['warm_start'],
                                     n_jobs=self.network_architecture['n_jobs'])
