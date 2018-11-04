import random
import numpy as np
import re


class MLP:
    def __init__(self, neurons_per_layer, transfer_fcn_per_layer, learning_rates_per_layer):
        """
        Construct a new MLP.
        The number of layers is given implicitly by the size of the lists.

        :param neurons_per_layer: A list of integers. Each integer sets the number of neurons for a layer.
        :param transfer_fcn_per_layer: A list of transfer functions, for each layer except the input layer.
        :param learning_rates_per_layer: A list of floats.
        """
        assert len(neurons_per_layer) == len(transfer_fcn_per_layer)
        random.seed(13579)

        self.learning_rates = learning_rates_per_layer
        self.transfer_functions = transfer_fcn_per_layer
        self.weight_matrix = []
        self.weight_update_matrix = []
        self.layer_error_matrix = []

        # Initialize the weight matrices for each layer with random values
        prev_neurons = 0
        for n in neurons_per_layer:
            if prev_neurons:  # Skip weights for input layer
                self.layer_error_matrix.append([0 for _ in range(n)])
                self.weight_matrix.append([[random.uniform(-2, 2) for _ in range(prev_neurons)] for _ in range(n)])
                self.weight_update_matrix.append([[0 for _ in range(prev_neurons)] for _ in range(n)])
            prev_neurons = n + 1  # +1 is for the bias

    @staticmethod
    def _calculate_layer_output(prev_output, layer_weights, transfer_function):
        """
        Calculate the output of every neuron for a given layer.

        :param prev_output: The output of the previous layer (or the input vector)
        :param layer_weights: The weights for this layer
        :param transfer_function: The transfer function to use for this layer.
        :return: A list with the output value of each neuron.
        """
        output = []
        for weights in layer_weights:
            weighted_sum = sum([x * w for x, w in zip(prev_output + [1], weights)])
            output.append(transfer_function.activate(weighted_sum))
        return output

    def _backprop_error(self, i, output, teacher):
        """
        Propagate the error of the output layer backwards through the network.
        The error gets stored in self.layer_error_matrix[<layer index>][<neuron index>]

        :param i: The index of the layer.
        :param output: The output matrix of the network.
        :param teacher: The wanted result.
        :return:
        """
        for j in range(len(self.weight_matrix[i])):
            # Error of the neurons in the output layer
            if len(self.weight_matrix) == i+1:
                self.layer_error_matrix[i][j] = (self.transfer_functions[i].derivative(output[i][j]) * (teacher[j] - output[i][j]))**2
            else:   # Error for the hidden layers
                x = 0
                for k in range(len(self.layer_error_matrix[i+1])):
                    x += self.layer_error_matrix[i+1][k] * self.weight_matrix[i+1][k][j]
                self.layer_error_matrix[i][j] = x * self.transfer_functions[i].derivative(output[i][j])

    def _update_layer_weights(self, i, output, input_vector):
        for j in range(len(self.weight_matrix[i])):
            for k in range(len(self.weight_matrix[i][j])):
                self.weight_matrix[i][j][k] += self.learning_rates[i] * self.layer_error_matrix[i][j] * output[i][j] / len(self.weight_matrix[i][j])

    def _backprop_step(self, input_vector, teacher_vector):
        prev_output = input_vector
        net_output = []

        # Forward through the net
        for transfer, weights in zip(self.transfer_functions, self.weight_matrix):
            prev_output = self._calculate_layer_output(prev_output, weights, transfer)
            net_output.append(prev_output)

        # Calculate the total error of the last layer, using quadratic differences
        total_error = sum([(x - y) ** 2 for x, y in zip(teacher_vector, net_output[-1])])

        # Calculate the error for each neuron, iterating backwards through the network
        for i in reversed(range(len(self.weight_matrix))):
            self._backprop_error(i, net_output, teacher_vector)

        # Update the weights
        for i in range(len(self.weight_matrix)):
            self._update_layer_weights(i, net_output, input_vector)

        return total_error

    def backpropagation(self, patterns, iterations):
        learning_curve = []
        for _ in range(iterations):
            total_error = 0
            random.shuffle(patterns)
            for input_vector, teacher_vector in patterns:
                total_error += self._backprop_step(input_vector, teacher_vector)
            learning_curve.append(total_error/len(patterns))

        return learning_curve

    def calculate_output(self, input_vector):
        prev_output = list(input_vector)

        # Forward through the net
        for transfer, weights in zip(self.transfer_functions, self.weight_matrix):
            prev_output = self._calculate_layer_output(prev_output, weights, transfer)

        net_output = prev_output
        return net_output


class IdentityTransfer:
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def derivative(x):
        return 1


class LogisticTransfer:
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(x))

    @classmethod
    def derivative(cls, x):
        z = cls.activate(x)
        return z * (1 - z)


class TanhTransfer:
    @staticmethod
    def activate(x):
        return np.tanh(x)

    @classmethod
    def derivative(cls, x):
        return 1 - cls.activate(x) ** 2


def read_patterns_from_file(filename):
    with open(filename, 'r') as fp:
        lines = list(fp)

    # Parse dimensions from the input file
    m = re.search(r'P=(?P<patterns>[0-9]+)[ ]*N=(?P<input>[0-9]+)[ ]*M=(?P<output>[0-9]+)', lines[1])
    in_size = int(m.group(2))
    out_size = int(m.group(3))
    patterns = []

    # Separate input vectors from teacher vectors
    for line in lines[2:]:
        l = list(filter(None, line[:-1].split(' ')))
        input_vector = [float(x) for x in l[:in_size]]
        teacher_vector = [float(x) for x in l[in_size:]]
        patterns.append((input_vector, teacher_vector))

    return in_size, out_size, patterns


def main():
    in_size, out_size, patterns = read_patterns_from_file('PA-B-train-04.dat')

    # Build the MLP and execute the backpropagation
    mlp = MLP([in_size, 5, 5, out_size], [LogisticTransfer(), LogisticTransfer(), LogisticTransfer(), LogisticTransfer()], [0.01, 0.01, 0.1])
    learning_curve = mlp.backpropagation(patterns, 500)

    # Save the learning curve
    with open('learning.curve', 'w') as fp:
        for error in learning_curve:
            fp.write("%s\n" % error)

    # Evaluate performance on test set
    _, _, test_patterns = read_patterns_from_file('PA-B-test-04.dat')

    test_error = 0
    for pattern, teacher in test_patterns:
        output = mlp.calculate_output(pattern)
        test_error += sum([(x - y) ** 2 for x, y in zip(teacher, output)])
    print("Average error of test set: ".format(test_error/len(test_patterns)))


if __name__ == "__main__":
    main()
