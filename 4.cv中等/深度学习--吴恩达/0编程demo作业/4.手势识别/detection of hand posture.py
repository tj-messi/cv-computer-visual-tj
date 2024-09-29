import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

#%matplotlib inline
np.random.seed(1)

def linear_function():
    """
    Implements a linear function: 
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- runs the function for Y = WX + b 
    """
    
    np.random.seed(1)
    
    # 初始化随机张量
    X = tf.constant(np.random.randn(3, 1), dtype=tf.float32, name="X")
    W = tf.constant(np.random.randn(4, 3), dtype=tf.float32, name="W")
    b = tf.constant(np.random.randn(4, 1), dtype=tf.float32, name="b")

    # 计算 Y = WX + b
    result = tf.add(tf.matmul(W, X), b)

    return result.numpy()  # 返回结果并转换为 NumPy 数组
'''
# 调用函数并打印结果
output = linear_function()
print(output)
'''

def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    
    # 计算 sigmoid(z)
    result = tf.sigmoid(z)

    return result.numpy()  # 返回结果并转换为 NumPy 数组

'''
z_input = tf.constant([0.0, 12], dtype=tf.float32)  # 示例输入
output = sigmoid(z_input)
print(output)  # 打印结果
'''


