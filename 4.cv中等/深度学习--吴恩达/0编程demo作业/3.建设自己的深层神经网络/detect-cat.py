import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# 加载数据集
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

'''
# Example of a picture
index = 7
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
plt.show()
'''

# 取出数据集的大小
m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
num_px = train_x_orig.shape[1]

'''
print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))
'''

# 统一重新调整图片大小
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# 标准化数据在0到1之间
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

'''
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
'''

# 设置神经网络的维度
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(layers_dims[0], layers_dims[1], layers_dims[2])
    ### END CODE HERE ###

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
        ### END CODE HERE ###

        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(A2, Y)
        ### END CODE HERE ###

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        ### END CODE HERE ###

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

'''
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
'''


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###

        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###

        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))

    return parameters



#parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


'''
## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "my_image1.jpg"  # 确保该图片在 "images" 目录下
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image

# 使用 PIL 调整图像大小
image = Image.open(fname)
num_px = 64  # 假设图像大小为 64x64
image = image.resize((num_px, num_px))  # 调整大小
my_image = np.array(image).reshape((1, num_px * num_px * 3)).T  # 确保形状正确，并转置

# 预测图像
my_label_y = [1]
my_predicted_image = predict(my_image, my_label_y, parameters)

# 假设 classes 是一个包含类名的列表
classes = ["cat", "non-cat"]

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
      classes[int(np.squeeze(my_predicted_image))] + "\" picture.")

# 显示图像
plt.imshow(image)
plt.axis('off')  # 隐藏坐标轴
plt.show()
'''

#pred_test = predict(test_x, test_y, parameters)

def random_layer_dims(n_layers, min_neurons, max_neurons):
    """
    随机生成每一层的神经元数量
    """
    return [np.random.randint(min_neurons, max_neurons) for _ in range(n_layers)]

def test_random_layer_dims(n_tests, n_layers, min_neurons, max_neurons, num_iterations=2500, print_cost=False):
    """
    测试随机生成的每一层神经元数量的模型准确率
    """
    accuracies = []
    for _ in range(n_tests):
        # 随机生成每一层的神经元数量
        hidden_layer_dims = random_layer_dims(n_layers, min_neurons, max_neurons)
        layers_dims = [n_x] + hidden_layer_dims + [n_y]

        # 训练模型
        parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=num_iterations, print_cost=print_cost)

        # 测试模型
        pred_test = predict(test_x, test_y, parameters)
        accuracy = np.mean(pred_test == test_y) * 100
        accuracies.append(accuracy)
        print(f"Test accuracy with random layer dimensions: {accuracy:.2f}% with parameters: {layers_dims}")

    return (layers_dims, accuracies)

# 测试随机生成的每一层神经元数量的模型准确率
n_tests = 5  # 测试次数
n_layers = 3  # 隐藏层层数
min_neurons = 5  # 每层最小神经元数量
max_neurons = 20 # 每层最大神经元数量

ans = test_random_layer_dims(n_tests, n_layers, min_neurons, max_neurons, num_iterations=2500, print_cost=True)
# 打印最大精确度和对应的参数
max_accuracy = max(ans[1])
max_index = ans[1].index(max_accuracy)
print(f"Maximum accuracy: {max_accuracy:.2f}% with parameters: {ans[0][max_index]}")