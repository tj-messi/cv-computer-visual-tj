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

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.random.set_seed(1)  # 设置随机种子
    
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.keras.initializers.GlorotUniform(seed=1)(shape=(25, 12288))
    b1 = tf.zeros((25, 1))
    W2 = tf.keras.initializers.GlorotUniform(seed=1)(shape=(12, 25))
    b2 = tf.zeros((12, 1))
    W3 = tf.keras.initializers.GlorotUniform(seed=1)(shape=(6, 12))
    b3 = tf.zeros((6, 1))
    ### END CODE HERE ###

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }
    
    return parameters


'''
# 创建参数
parameters = initialize_parameters()
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

import tensorflow as tf

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                       # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                      # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                      # Z3 = np.dot(W3, A2) + b3
    ### END CODE HERE ###
    
    return Z3

def create_placeholders(input_size, num_examples):
    """
    创建输入与目标的占位符
    """
    X = tf.zeros((input_size, num_examples))  # 输入占位符
    Y = tf.zeros((6, num_examples))             # 目标占位符
    return X, Y

'''
parameters = initialize_parameters()
X, Y = create_placeholders(12288, 6)
Z3 = forward_propagation(X, parameters)

print("Z3 = " + str(Z3.numpy()))  # 使用 .numpy() 方法获取 NumPy 数组格式的输出
'''


def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the TensorFlow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    ### END CODE HERE ###
    
    return cost


'''
# 创建参数和输入数据
parameters = initialize_parameters()
X, Y = create_placeholders(12288, 6)
Z3 = forward_propagation(X, parameters)

# 计算成本
cost = compute_cost(Z3, Y)

print("cost = " + str(cost.numpy()))  # 使用 .numpy() 方法获取 NumPy 数组格式的输出
'''

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer TensorFlow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    tf.random.set_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m: number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y: output size
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X = tf.keras.Input(shape=(n_x,))
    Y = tf.keras.Input(shape=(n_y,))
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ### END CODE HERE ###

    # Training loop
    for epoch in range(num_epochs):
        epoch_cost = 0.                       # Defines a cost related to an epoch
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            # IMPORTANT: The line that runs the graph on a minibatch.
            with tf.GradientTape() as tape:
                # Forward propagation
                Z3 = forward_propagation(minibatch_X, parameters)
                minibatch_cost = compute_cost(Z3, minibatch_Y)
            
            # Backpropagation
            grads = tape.gradient(minibatch_cost, parameters.values())
            optimizer.apply_gradients(zip(grads, parameters.values()))
            
            epoch_cost += minibatch_cost.numpy() / num_minibatches

        # Print the cost every epoch
        if print_cost and epoch % 100 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost and epoch % 5 == 0:
            costs.append(epoch_cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Save the parameters
    print("Parameters have been trained!")

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(Z3, axis=1), tf.argmax(Y, axis=1))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Train Accuracy:", accuracy.numpy())
    print("Test Accuracy:", accuracy.numpy())

    return parameters

# 需要创建 X_train, Y_train, X_test, Y_test 数据
parameters = model(X_train, Y_train, X_test, Y_test)
