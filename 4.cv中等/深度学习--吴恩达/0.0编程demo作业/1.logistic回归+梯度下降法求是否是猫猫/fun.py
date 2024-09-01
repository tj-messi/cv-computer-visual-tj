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

def L1_Loss(y_hat,y):
    #L1损失函数
    return np.sum(np.abs(y_hat-y))

def L2_Loss(y_hat,y):
    #L2损失函数
    return np.sum(np.abs(y_hat-y)**2);

def initialize_with_zeros(dim):
    # 初始化为0向量
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b

def propagate(w, b, X, Y):
    # 前向\后向传播
    '''
    w：权重矩阵，形状通常为 (n_x, 1)，其中 n_x 是输入特征的数量。
    b：偏置项，一个标量。
    X：输入数据矩阵，形状为 (n_x, m)，其中 m 是样本数量。
    Y：目标输出矩阵，形状为 (1, m)，包含了每个样本的真实标签。
    '''
    m = X.shape[1]
    A = basic_sigmoid(np.dot(w.T, X) + b)  # 计算激活值
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # 计算代价
    dw = 1 / m * np.dot(X, (A - Y).T)  # 计算w的梯度
    db = 1 / m * np.sum(A - Y)  # 计算b的梯度

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    # 优化函数
    costs = []


    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))


    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs
def predict(w, b, X):
    # 预测函数
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = basic_sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


