# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:24:16 2018

@author: Maren
"""

import numpy as np
import sys


class ROLF:
    """
    Initialize a ROLF network

    eta_sigma, eta_c: parameters for adapting the center and the perceptive
        area of a neuron
    init_sigma: initial value for sigma
    init_method: how to initialize the sigma values of new neurons ("init" or "mean")
    rho: parameter to determine the size of the extended perceptive area
        of a neuron
    max_num_neurons, max_num_patterns: maximum number of neurons/patterns
    """

    def __init__(self, eta_sigma, eta_c, init_sigma, init_method, rho=0.5, max_num_neurons=1000,
                 max_num_patterns=100000, max_input_dim=10):
        self.eta_sigma = eta_sigma
        self.eta_c = eta_c
        self.init_sigma = init_sigma
        self.init_method = init_method
        self.rho = rho
        self.max_num_neurons = max_num_neurons
        self.max_num_patterns = max_num_patterns
        self.max_input_dim = max_input_dim

        self.neurons = []

    # implements the Euclidean distance
    def dist(self, x1, x2):
        distances_squared = [(x1[i] - x2[i]) * (x1[i] - x2[i]) for i in range(len(x1))]
        return np.sqrt(np.sum(distances_squared))

    def train(self, patterns):
        # check for consistency with input_dim
        if len(patterns) > self.max_num_patterns:
            print("Number of patterns exceeds maximum! Exiting")
            sys.exit(1)

        if len(patterns[0]) > self.max_input_dim:
            print("Input dimension exceeds maximum! Exiting")
            sys.exit(1)

        for pattern in patterns:
            # determine winning neuron and adapt parameters
            accepting_neuron = None
            for neuron in self.neurons:
                # check if pattern lies within perceptive area
                if self.dist(pattern, neuron.center_pos) <= neuron.sigma * self.rho:
                    # is it the closest neuron found so far?
                    if accepting_neuron is None or self.dist(pattern, neuron.center_pos) < self.dist(pattern,
                                                                                                     accepting_neuron.center_pos):
                        accepting_neuron = neuron

            if accepting_neuron is not None:
                accepting_neuron.train(pattern)

            elif len(self.neurons) < self.max_num_neurons:
                if self.init_method == "mean" and len(self.neurons) != 0:
                    sigma = np.mean([neuron.sigma for neuron in self.neurons])
                else:
                    sigma = self.init_sigma
                self.neurons.append(self.neuron(pattern, sigma, self.eta_sigma, self.eta_c, self))

            # when training is done: connect the perceptive areas, and assign the patterns to clusters

    class neuron:
        """
        Initialize a new ROLF neuron with the given initialization method.

        center_pos: position of the center in input space
        sigma: size of the perceptive area
        """

        def __init__(self, center_pos, sigma, eta_sigma, eta_c, rolf):
            self.center_pos = center_pos
            self.sigma = sigma
            self.eta_sigma = eta_sigma
            self.eta_c = eta_c
            self.rolf = rolf

        '''
        Train a neuron with a new training pattern, and update the center position
        and the perceptive area.
        '''

        def train(self, pattern):
            dist = self.rolf.dist(pattern, self.center_pos)
            dist_vector = [abs(x1 - x2) for (x1, x2) in zip(pattern, self.center_pos)]

            self.center_pos = [self.eta_c * (x1 + x2) for (x1, x2) in zip(self.center_pos, dist_vector)]
            self.sigma = self.sigma + self.eta_sigma * (dist - self.sigma)


def main():
    eta_sigma = 0.05
    eta_c = 0.05
    init_sigma = 0.04
    init_method = "mean"

    with open('training_2dim.dat', 'r') as fp:
        lines = list(fp)

    # Parse dimensions from the input file
    patterns = []

    # Read input vectors
    i = 0
    for line in lines:
        if i == 0:
            in_size = len(list(filter(None, line.split(' '))))
        l = list(filter(None, line.split(' ')))
        input_vector = [float(x) for x in l[:in_size]]
        patterns.append(input_vector)
        i += 1

    # initialize a ROLF network and train it on the given input
    rolf = ROLF(eta_sigma, eta_c, init_sigma, init_method)
    rolf.train(patterns)

    # write output to file
    with open("output_2dim.txt", "w") as fp:
        fp.write("# Results for initial sigma = {0}, eta_c = {1}, eta_sigma = {2}, \"{3}\" method\n\n".format(init_sigma,
                                                                                                            eta_sigma,
                                                                                                            eta_c,
                                                                                                            init_method))
        fp.write("# Neuron center positions{0}Neuron sizes\n".format("".join(["\t" for _ in range(in_size - 2)])))
        for neuron in rolf.neurons:
            fp.write("{0}\t\t{1}\n".format("\t".join([str(round(f, 3)) for f in neuron.center_pos]), neuron.sigma))


if __name__ == "__main__":
    main()
