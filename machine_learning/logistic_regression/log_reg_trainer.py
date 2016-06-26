from sklearn import metrics


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

    def confusion_matrix(self, X_test, Y_test):
        """
        Confusion matrix for logistic regression

        :param X_test:
        :param Y_test:
        :return:
        """
        predicted = self.logistic_regression.predict(X_test)
        report = metrics.classification_report(Y_test, predicted)
        confusion_matrix = metrics.confusion_matrix(Y_test, predicted)
        return report, confusion_matrix

    def regression_coefficients(self):
        """
        Current state of coefficients

        :return:
        """
        return self.logistic_regression.coef_
