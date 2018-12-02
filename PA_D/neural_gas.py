from math import sqrt, exp
from operator import itemgetter
import random

random.seed(10001)


class NeuralGas:
    def __init__(self, dimensions, num_neurons, gaussian_width):
        self.centers = [[random.uniform(0, 1) for _ in range(dimensions)] for _ in range(num_neurons)]

        # Adjusts the width of the gaussian
        self.s = gaussian_width

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

    def apply_learning_rule(self, stimulus, learning_rate):
        """
        Updates the centers for a given stimulus.
        :return:
        """
        # Get a list of the indexes of the sorted neurons
        neuron_list = [x[0] for x in sorted(enumerate(self.cached_responses), key=lambda x: x[1])]

        # Calculate and apply delta C_j. The distance is the position in the sorted neuron list.
        for dist, index in enumerate(neuron_list):
            # Calculate the delta
            delta = [learning_rate * self._neighborhood_fcn(dist) * (s - c)
                     for s, c in zip(stimulus, self.centers[index])]

            # Apply it to the center
            for i in range(len(delta)):
                self.centers[index][i] += delta[i]

    def _neighborhood_fcn(self, dist):
        return exp(-1 / 2 * (dist ** 2) / (self.s ** 2))


class MultiNeuralGas:
    def __init__(self, partner_networks, dimensions, neurons, lrate_max, lrate_min, gaussian_width):
        """

        :param partner_networks: Number of partner networks.
        :param dimensions: Dimensions of the patterns
        :param neurons: Number of neurons to use per net
        :param lrate_max: The learning rate in the beginning
        :param lrate_min: The learning rate at the end
        """
        self.max = lrate_max
        self.min = lrate_min
        self.partners = [NeuralGas(dimensions, neurons, gaussian_width) for _ in range(partner_networks)]

    def train(self, patterns, iterations):
        """
        Trains the network on the given patterns.

        :param patterns:
        :param iterations:
        """

        for t in range(iterations):
            random.shuffle(patterns)

            for stimulus in patterns:
                self._training_step(stimulus, t, iterations)

    def _training_step(self, stimulus, t, iterations):
        """
        Executes a single step of learning by processing a single pattern.
        :param stimulus:
        :param t: The current timestep
        :param iterations: Total number of iterations. Needed for the dynamic learning rate.
        :return:
        """

        # Calculate the minimal responses of all networks
        responses = []
        for network in self.partners:
            response = network.calculate_responses(stimulus)
            responses.append((response, network))

        # Select the network with the minimal response and update its weights
        winner = min(responses, key=itemgetter(0))[1]
        winner.apply_learning_rule(stimulus, self.learning_rate(t, iterations))

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

    def learning_rate(self, t, iterations):
        return self.max * ((self.max / self.min) ** (-t / iterations))


def training_file():
    """
    Trains the network with the patterns from PA-D-train.dat
    """
    mngas = MultiNeuralGas(4, 2, 30, lrate_max=0.01, lrate_min=0.001, gaussian_width=6)
    mngas.store_centers('initial.net')

    with open("PA-D-train.dat", 'r') as fp:
        lines = list(fp)

    patterns = []
    for line in lines[2:]:
        p = [float(x) for x in filter(None, line[:-1].split(' '))]
        patterns.append(p)

    mngas.train(patterns, 120)
    mngas.store_centers('PA-D.net')


def training_random():
    """
    Trains the network with patterns drawn uniformly from 3 non-overlapping areas.
    The boundaries can be read from the initialization.
    """
    patterns = []
    patterns += [[random.uniform(0.8, 1), random.uniform(0.1, 0.25)] for _ in range(300)]
    patterns += [[random.uniform(0.05, 0.15), random.uniform(0.8, 0.95)] for _ in range(300)]
    patterns += [[random.uniform(0.6, 0.8), random.uniform(0.6, 0.8)] for _ in range(300)]

    mngas = MultiNeuralGas(4, 2, 30, lrate_max=0.01, lrate_min=0.001, gaussian_width=1.5)
    mngas.store_centers('initial.net')
    mngas.train(patterns, 1000)
    mngas.store_centers('PA-D.net')


if __name__ == "__main__":
    training_file()
    #training_random()

