# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:11:37 2018

@author: Maren
"""

import random

if __name__ == '__main__':
    numPatterns = 1000
    dim = 2

    patterns = []
    patterns += [[random.uniform(0.8, 1), random.uniform(0.1, 0.25)] for _ in range(300)]
    patterns += [[random.uniform(0.05, 0.15), random.uniform(0.8, 0.95)] for _ in range(300)]
    patterns += [[random.uniform(0.6, 0.8), random.uniform(0.6, 0.8)] for _ in range(300)]

    with open("training_2dim.dat", "w") as fp:
        for pattern in patterns:
            fp.write((" ".join([str(num) for num in pattern])))
            fp.write("\n")

