# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:44:58 2018

@author: Maren

A python class to model a 2 layer Perzeptron.

"""
import sys
import numpy as np
import pprint


class Perzeptron:
    """
    Class initialization
        maxN: maximum dimensionality of input vector x
        maxM: maximum dimensionality of output vector y
        maxP: maximum number of training examples
    """
    def __init__(self, maxN=101, maxM=30, maxP=200, learning_rate=0.5):
        self.output = []
        self.learning_rate = learning_rate
        self.maxN = maxN
        self.maxM = maxM
        self.maxP = maxP
        self.weight_matrix = None
        self.X = None
        self.Y = None
        
    # for a scalar input value, returns the output of the logistic function ( 1 / 1 + exp(-x))
    def logistic_function(self, x):
        return 1 / (1 + np.exp(x))
        
    '''
    read the input data from a file
    Assumes the data to be in the format:
        x1 x2 x3 ... : y1 y2 ...
    (x and y separated by ":", one line per training example)
    '''
    def read_input_data(self, filename):
        with open(filename, "r") as fp:
            data = fp.read()

        X = []
        Y = []
        
        for line in data.split("\n"):
            x = [float(val) for val in line.split("\t")[0].split(" ") if val != ""]
            y = [float(val) for val in line.split("\t")[1].split(" ") if val != ""]

            X.append(x)
            Y.append(y)
            
        self.X = np.array(X)
        self.Y = np.array(Y)
            
        if self.X.shape[0] != self.Y.shape[0]:
            print("Dimension mismatch! Exit")
            sys.exit(1)
        if self.X.shape[1] > self.maxN or self.X.shape[1] < 1:
            print("Invalid dimensionality of input! Choose 0 < N < 101")
            sys.exit(1)
        if self.Y.shape[1] > self.maxM or self.Y.shape[1] < 1:
            print("Invalid dimensionality of output! Choose 0 < M < 30")
            sys.exit(1)
        if self.X.shape[0] > self.maxP or self.X.shape[0] < 1:
            print("Invalid dimensionality of input! Choose 0 < P < 200")
            sys.exit(1)

    def read_weights_from_file(self, filename):
        with open(filename, "r") as fp:
            data = fp.read()
        weight_matrix = []
        for line in data.split("\n"):
            weight_matrix.append([float(val) for val in line.split(" ") if val != ""])

        self.weight_matrix = np.array(weight_matrix)

    # training the perzeptron on training data matrix X and label matrix Y
    def train(self, weights_filename=None):
        if weights_filename is not None:
            self.read_weights_from_file(weights_filename)
        else:
            # initialize weight matrix randomly and normalize weights to [0,0.5]
            self.weight_matrix = np.random.rand(self.X.shape[1]+1, self.Y.shape[1])
            self.weight_matrix /= 2
        
        # update weights
        for i in range(self.X.shape[0]):
            for j in range(self.Y.shape[1]):
                x_i_with_bias = np.concatenate((self.X[i], [1.0]))
                weighted_sum = sum([x_i_with_bias[k]*self.weight_matrix[k][j] for k in range(len(x_i_with_bias))])
                # print(weighted_sum)
                neuron_output = self.logistic_function(weighted_sum)
                # print(neuron_output)
                for l in range(len(x_i_with_bias)):
                    # perzeptron training rule
                    weight_update = (-1.0)*(self.learning_rate*(self.Y[i][j]-neuron_output)*x_i_with_bias[l])
                    self.weight_matrix[l][j] += weight_update
        
        # calculate final perzeptron output
        for i in range(self.X.shape[0]):
            partial_output = []
            for j in range(self.Y.shape[1]):
                x_i_with_bias = np.concatenate((self.X[i], [1.0]))
                weighted_sum = sum([x_i_with_bias[k]*self.weight_matrix[k][j] for k in range(len(x_i_with_bias))])
                neuron_output = self.logistic_function(weighted_sum)
                partial_output.append(neuron_output)
            self.output.append(partial_output)
            
    # test the perzeptron on an unknown pattern x_test
    def test(self, x_test):
        output = []
        for j in range(self.Y.shape[1]):
            x_i_with_bias = np.concatenate((x_test, [1.0]))
            weighted_sum = sum([x_i_with_bias[k]*self.weight_matrix[k][j] for k in range(len(x_i_with_bias))])
            neuron_output = self.logistic_function(weighted_sum)
            output.append(neuron_output)
        return output


def main():
    perzeptron = Perzeptron(learning_rate=0.75)
    perzeptron.read_input_data("PA-A-train.dat")
    perzeptron.train()

    # To test with weights from a file, uncomment this
    #perzeptron.train(weights_filename='weights.dat')
    
    print("Weight matrix after training:\n")
    print(perzeptron.weight_matrix)
    
    print("\nFinal perzeptron output for the training data:\n")
    pprint.pprint(perzeptron.output)


if __name__ == "__main__":
    main()
