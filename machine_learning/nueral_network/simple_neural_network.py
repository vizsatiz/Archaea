import numpy as num_py


class CostAndGradient:
    def __init__(self, hidden_layers, input_size, number_of_output_classes):
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.number_of_output_classes = number_of_output_classes
        self.array_of_thetas = self.initialize_theta(hidden_layers)

    @staticmethod
    def forward_propogation(self, x, y, theta_list):
        return

    def initialize_theta(self, hidden_layers, input_size):
        theta_list = []
        number_of_nodes_in_previous_layer = input_size
        number_of_hidden_layers = len(hidden_layers)
        for number_of_nodes in hidden_layers:
            theta_matrix = self.initialize_theta_matrix(number_of_nodes_in_previous_layer, number_of_nodes)
            theta_list.append(theta_matrix)
            number_of_nodes_in_previous_layer = number_of_nodes
        return theta_list

    def initialize_theta_matrix(self, rows, columns):
        return num_py.random.random((rows, columns)) - 1
