# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:49:58 2024

@author: Michal
"""

import numpy as np

def square_root_function(x, a, b):
    return np.sqrt(a * x) + b

def straight_line_function(x, m, c):
    return m * x + c

def logistic_function(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

def quadratic_function(x, a, b, c):
    return a * (x - c)**2 + b

def fifth_order(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f