import sklearn.metrics as matrics


class ConvNeuralNetTrainer:

    def __init__(self, network):
        self.network = network

    def train(self, X_train, Y_train):
        """
        The method trains the convolutional net with data

        :param X_train: The training set input value
        :param Y_train: Training expected outputs
        :return:
        """
        return self.network.fit(X_train, Y_train)

    def predict(self, X_prediction):
        """
        This method makes prediction based on trained network

        :param X_prediction:
        :return:
        """
        return self.network.predict(X_prediction)

    def confusion_matrix(self, X_test, Y_test):
        """
        Gets the error across given input

        :param X_test:
        :param Y_test:
        :return:
        """
        prediction = self.network.predict(X_test)
        report = matrics.classification_report(Y_test, prediction)
        confusion_matric = matrics.confusion_matrix(Y_test, prediction)
        return report, confusion_matric

