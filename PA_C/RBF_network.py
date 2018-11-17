# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:36:15 2018

@author: Maren
"""

import random
import numpy as np
import re
import sys
from pprint import pprint

'''
This RBF network implements single step learning.
The RBF centers are initialized in a data driven manner (set a center for
each point in a subset of the training data, chosen randomly).
The widths are initialized randomly, with a value between 0 and 0.5 .
'''

class RBF_network:
    def __init__(self,num_rbf_neurons, patterns, learning_rate=0.1):
        '''
        Initialize a new RBF network.
        
        num_rbf_neurons: number of neurons in the RBF layer
        X: input patterns 
        Y: teacher values
        learning_rate: learning rate for all output neurons
        
        The number of input and output neurons is implicitly given by the
        size of input and teacher vector.
        '''
        random.seed(10001)
        
        self.learning_rate = learning_rate
        self.input_size = len(patterns[0][0])
        self.output_size = len(patterns[0][1])
        self.num_rbf_neurons = num_rbf_neurons

        # Initialize the weight matrix with random values
        self.weight_matrix = [[random.uniform(-0.5, 0.5) for _ in range(self.output_size)] for _ in range(num_rbf_neurons+1)]
        
        # Initialize RBF centers and widths
        self.rbf_centers = []
        self.rbf_widths = [random.uniform(0.0, 2.0) for _ in range(num_rbf_neurons)]    
        #self.rbf_widths = [0.8 for _ in range(num_rbf_neurons)]
        for k in range(num_rbf_neurons):
            self.rbf_centers.append(random.choice(patterns)[0])
            #self.rbf_centers.append([random.uniform(0.0,1.0) for dim in range(self.input_size)])
            
    def gaussian_transfer_function(self,input_vector, center_vector, width):
        sum_of_squares = np.sum([(x-c)*(x-c) for x,c in zip(input_vector, center_vector)])
        euclidean_distance = np.sqrt(sum_of_squares)
        return np.exp((-1)*euclidean_distance/2*width*width)
    
    def train(self, patterns, num_iterations):
        X = np.array([pattern[0] for pattern in patterns])
        Y = np.array([pattern[1] for pattern in patterns])
        
        if X.shape[1] != self.input_size or Y.shape[1] != self.output_size:
            print("Dimension mismatch! Exiting")
            sys.exit(1)
        
        learning_curve = []
        for n in range(num_iterations):
            total_error = 0
            for i in range(X.shape[0]):
                pattern_error = []
                for j in range(Y.shape[1]):
                    # calculate output for the RBF neurons
                    rbf_output = []
                    for k in range(self.num_rbf_neurons):
                        rbf_output.append(self.gaussian_transfer_function(X[i],self.rbf_centers[k],self.rbf_widths[k]))
                    rbf_output_with_bias = np.concatenate((rbf_output, [1.0]))
                    weighted_sum = sum([rbf_output_with_bias[k]*self.weight_matrix[k][j] for k in range(len(rbf_output_with_bias))])
                    pattern_error.append((Y[i]-weighted_sum) ** 2)
                    # print(neuron_output)
                    for l in range(len(rbf_output_with_bias)):
                        # perzeptron training rule
                        weight_update = (self.learning_rate*(Y[i][j]-weighted_sum)*rbf_output_with_bias[l])
                        self.weight_matrix[l][j] += weight_update
                pattern_error = np.sum(pattern_error)
                total_error += pattern_error
            #print(total_error)
            learning_curve.append(total_error)
            
        return learning_curve
    
    def calculate_output(self, test_vector):   
        if len(test_vector) != self.input_size:
            print("Dimension mismatch! Exiting")
            sys.exit(1)
            
        X = np.array(test_vector)        
        output = []
        
        for j in range(self.output_size):
            rbf_output = []
            for k in range(self.num_rbf_neurons):
                rbf_output.append(self.gaussian_transfer_function(X,self.rbf_centers[k],self.rbf_widths[k]))
            rbf_output_with_bias = np.concatenate((rbf_output, [1.0]))
            weighted_sum = sum([rbf_output_with_bias[k]*self.weight_matrix[k][j] for k in range(len(rbf_output_with_bias))])
            output.append(weighted_sum)
             
        return output
        
    
def main():
    with open('training.dat', 'r') as fp:
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
        patterns.append((input_vector,teacher_vector))

    # Build the MLP and execute the backpropagation
    rbf_network = RBF_network(2, patterns)
    learning_curve = rbf_network.train(patterns, 100)

    # Save the learning curve
    with open('learning.curve', 'w') as fp:
        for error in learning_curve:
            fp.write("%s\n" % error)

    # read test data
    with open('test.dat', 'r') as fp:
        lines = list(fp)

    test_patterns = []
    for line in lines[1:]:
        l = list(filter(None, line[:-1].split(' ')))
        input_vector = [float(x) for x in l[:in_size]]
        test_patterns.append(input_vector)

    for pattern in test_patterns:
        print("Network output for test pattern {0}:".format(pattern))
        print(rbf_network.calculate_output(pattern))

    
if __name__ == "__main__":
    main()
