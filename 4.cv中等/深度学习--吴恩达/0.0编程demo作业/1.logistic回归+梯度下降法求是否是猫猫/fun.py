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


def normalize_rows(x):
    #归一化数据
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def softmax(x):
    #softmax函数
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

'''
def vect2image(vector, shape):
    #将向量转化为图像
    return vector.reshape(shape)
'''

def L1_Loss(y_hat,y):
    #L1损失函数
    return np.sum(np.abs(y_hat-y))

def L2_Loss(y_hat,y):
    #L2损失函数
    return np.sum(np.abs(y_hat-y)**2);





