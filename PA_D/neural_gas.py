from math import sqrt, exp
from operator import itemgetter
import random

random.seed(10001)


class NeuralGas:
    def __init__(self, dimensions, num_neurons):
        self.centers = [[random.uniform(0, 1) for _ in range(dimensions)] for _ in range(num_neurons)]

        self.s = 2

        # This holds the calculated responses for the last applied stimulus
        self.cached_responses = []

    def calculate_responses(self, stimulus):
        """
        Calculates the responses for a given stimulus and caches them, for further steps.

        :param stimulus: An input vector
        :return: The response of the winner neuron
        """
        self.cached_responses = []

        for center in self.centers:
            # Using the euclidean distance as the response
            distance = sqrt(sum([(x - y) ** 2 for x, y in zip(stimulus, center)]))
            self.cached_responses.append(distance)

        return min(self.cached_responses)

    def apply_learning_rule(self, stimulus, t):
        # Get a list of the indexes of the sorted neurons
        neuron_list = [x[0] for x in sorted(enumerate(self.cached_responses), key=lambda x: x[1])]

        # Calculate and apply delta C_j. The distance is the position in the sorted neuron list.
        for dist, index in enumerate(neuron_list):
            # Calculate the delta
            delta = [self._learning_rate(t) * self._neighborhood_fcn(dist) * (s - c)
                     for s, c in zip(stimulus, self.centers[index])]

            # Apply it to the center
            for i in range(len(delta)):
                self.centers[index][i] += delta[i]

    def _neighborhood_fcn(self, dist):
        return exp(-1 / 2 * (dist ** 2) / (self.s ** 2))

    def _learning_rate(self, t):
        return 0.01


class MultiNeuralGas:
    def __init__(self, partner_networks, dimensions, neurons):
        self.partners = [NeuralGas(dimensions, neurons) for _ in range(partner_networks)]

    def train(self, patterns, iterations):
        for t in range(iterations):
            random.shuffle(patterns)

            for stimulus in patterns:
                self._training_step(stimulus, t)

    def _training_step(self, stimulus, t):
        # Calculate the minimal responses of all networks
        responses = []
        for network in self.partners:
            response = network.calculate_responses(stimulus)
            responses.append((response, network))

        # Select the network with the minimal response and update its weights
        winner = min(responses, key=itemgetter(0))[1]
        winner.apply_learning_rule(stimulus, t)

    def store_centers(self, filename):
        """
        Writes the centers to a file, using the format specified in the exercise.

        :param filename:
        :return:
        """
        with open(filename, 'w') as fp:
            for network in self.partners:
                for center in network.centers:
                    fp.write('  '.join([str(x) for x in center]))
                    fp.write('\n')


def read_patterns_from_file(filename):
    with open(filename, 'r') as fp:
        lines = list(fp)

    patterns = []
    for line in lines[2:]:
        p = [float(x) for x in filter(None, line[:-1].split(' '))]
        patterns.append(p)

    return patterns


def main():
    mngas = MultiNeuralGas(4, 2, 20)
    mngas.store_centers('initial.net')

    patterns = read_patterns_from_file('PA-D-train.dat')
    mngas.train(patterns, 120)
    mngas.store_centers('PA-D.net')


if __name__ == "__main__":
    main()
