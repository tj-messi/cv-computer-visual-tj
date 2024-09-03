import numpy as np
import planar_utils
import testCases
import scipy
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]  # size of input layer
    n_h = 4
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)

'''
X_assess, Y_assess = testCases.layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))
'''
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    ### END CODE HERE ###

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
'''
n_x, n_h, n_y = testCases.initialize_parameters_test_case()

parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

def sigmoid(z):
    return 1/(1+scipy.special.expit(-z))

def forward_propagation(X, parameters):
    """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
    # 从parameters中取出W1, b1, W2, b2
    W1 = parameters.get('W1')
    b1 = parameters.get('b1')
    W2 = parameters.get('W2')
    b2 = parameters.get('b2')
    A0 = X

    # 前向传播计算
    Z1 = np.dot(W1, A0) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    #设定A2为列向量输出
    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

'''
X_assess, parameters = testCases.forward_propagation_test_case()

A2, cache = forward_propagation(X_assess, parameters)

# Note: we use the mean here just to make sure that your output matches ours.
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
'''

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    # 获取样本数量
    m = Y.shape[1]

    # 获取w1，w2
    W1 = parameters.get('W1')
    W2 = parameters.get('W2')

    # 成本计算
    epsilon = 1e-15  # 添加一个很小的常数
    logprobs = np.multiply(np.log(A2+epsilon), Y) + np.multiply(np.log(1 - A2+epsilon), (1 - Y))
    cost = np.sum(logprobs, axis=1, keepdims=True) / -m

    #使用np.squeeze去除成本数组中的单维度条目，确保成本是一个标量（即单个浮点数）
    cost = np.squeeze(cost)
    cost = float(cost)
    assert (isinstance(cost, float)) # 检查cost类型

    return cost

'''
A2, Y_assess, parameters = testCases.compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))
'''


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # 取出w1.w2
    W1 = parameters.get('W1')
    W2 = parameters.get('W2')

    # 取出A1.A2
    A1 = cache.get('A1')
    A2 = cache.get('A2')

    # 计算 dZ2,dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    # A1=tanh(Z1) 所以 g'(Z1)=(1-np.power(A1,2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads
'''
parameters, cache, X_assess, Y_assess =testCases. backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))
'''


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # 从参数中获得w1,b1,w2,b2
    W1 = parameters.get('W1')
    b1 = parameters.get('b1')
    W2 = parameters.get('W2')
    b2 = parameters.get('b2')

    # 从梯度中获得dW1,db1,dW2,db2
    dW1 = grads.get('dW1')
    db1 = grads.get('db1')
    dW2 = grads.get('dW2')
    db2 = grads.get('db2')

    # 更新参数
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
'''
parameters, grads = testCases.update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(3)
    # layer_sizes(X, Y)返回输入层，隐藏层，输出层的大小.这里取0就是x的大小，取2就是y的大小
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # 初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters.get('W1')
    b1 = parameters.get('b1')
    W2 = parameters.get('W2')
    b2 = parameters.get('b2')

    # 循环迭代num_iterations次
    for i in range(0, num_iterations):
        # 前向传播计算
        A2, cache = forward_propagation(X, parameters)

        # 成本计算
        cost = compute_cost(A2, Y, parameters)

        # 反向传播计算梯度
        grads = backward_propagation(parameters, cache, X, Y)

        # 梯度下降更新参数
        parameters = update_parameters(parameters, grads)

        # 每1000次打印一次成本
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters
'''
X_assess, Y_assess = testCases.nn_model_test_case()

parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

