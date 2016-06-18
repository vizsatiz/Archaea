class LogisticRegressionTrainer:

    def __init__(self, logistic_regression):
        self.logistic_regression = logistic_regression

    def train(self, X_train, Y_train):
        """
        Train the logistic regression

        :param X_train:
        :param Y_train:
        :return:
        """

        self.logistic_regression.fit(X_train, Y_train)

    def predict(self, X):
        """
        Predict the output of prediction with threshold set in __init__

        :param X:
        :param theta:
        :return:
        """
        return self.logistic_regression.predict(X)