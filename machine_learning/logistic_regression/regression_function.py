import math as math


def sigmoid(X):
    denominator = 1.0 + math.e ** (-1.0 * X)
    sigmoid_value = 1.0 / denominator
    return sigmoid_value
