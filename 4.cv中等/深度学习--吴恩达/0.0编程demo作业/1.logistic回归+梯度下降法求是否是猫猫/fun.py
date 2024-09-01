import math
import numpy as np

def basic_sigmoid(x):
    #基础basic函数
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    #sigmoid函数函数
    s = basic_sigmoid(x)
    ds = s * (1 - s)
    return ds

def image2vector(image):
    #将图像转化为向量
    return image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))




