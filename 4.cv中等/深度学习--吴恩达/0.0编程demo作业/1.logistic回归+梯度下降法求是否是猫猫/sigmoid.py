import math
import numpy as np

def basic_sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([1, 2, 3])
print(basic_sigmoid(x))

def sigmoid_derivative(x):
    
