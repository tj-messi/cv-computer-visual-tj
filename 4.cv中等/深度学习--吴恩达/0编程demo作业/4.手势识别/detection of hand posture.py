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

def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Returns:
    cost -- computed cost using sigmoid cross entropy
    """
    
    # 使用 sigmoid cross entropy 计算成本
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    
    # 返回平均成本
    return cost.numpy()  

'''
# 示例使用
logits_input = tf.constant(sigmoid([0.2,0.4,0.7,0.9]), dtype=tf.float32)  # 示例 logits
labels_input = tf.constant([0,0,1,1], dtype=tf.float32)   # 示例 labels
cost_value = cost(logits_input, labels_input)
print ("cost = " + str(cost_value))
'''

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
    corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
    will be 1. 
    
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    # 使用 tf.constant 创建深度 C
    # 注意这里 C 是用来表示类别数，不需要重新创建
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    
    # 返回 one hot 矩阵并转换为 NumPy 数组
    return one_hot_matrix.numpy()

'''
labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = " + str(one_hot))
'''

def ones(shape):
    """
    Creates an array of ones of dimension shape
    
    Arguments:
    shape -- shape of the array you want to create
        
    Returns: 
    ones -- array containing only ones
    """
    
    # 使用 tf.ones 创建全为 1 的张量
    ones = tf.ones(shape)
    
    # 直接返回 NumPy 数组
    return ones.numpy()

'''
print ("ones = " + str(ones([3])))
'''

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

'''
# Example of a picture
index = 80
plt.imshow(X_train_orig[index])
plt.show()
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
'''

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)


'''
print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
'''
