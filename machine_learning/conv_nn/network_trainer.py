class cnn_trainer:

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