# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:11:37 2018

@author: Maren
"""

import numpy as np

numPatterns = 100
dim = 9

patterns = np.random.rand(numPatterns, dim)

with open("training3.dat", "w") as fp:
    for i in range(numPatterns):
        fp.write((" ".join([str(num) for num in patterns[i]])))
        fp.write("\n")