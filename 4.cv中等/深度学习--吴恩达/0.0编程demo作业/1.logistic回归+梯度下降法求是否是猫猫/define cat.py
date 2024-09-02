import numpy as np
import matplotlib.pyplot as plt
import imageio
import h5py
import scipy
import fun
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# 加载数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 预处理图片的几个数据：训练数量、测试数量、图片高度和宽度
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# 转换numpy数组为展开的(num_px * num_px * 3, 1)的矩阵
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


# 归一化数据
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = fun.initialize_with_zeros(X_train.shape[0])
    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = fun.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = fun.predict(w, b, X_test)
    Y_prediction_train = fun.predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "my_image.jpg"  # 确保该图片在 "images" 目录下
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image

# 使用 imageio 读取图像
image = imageio.imread(fname)

# 使用 PIL 调整图像大小
image = Image.fromarray(image)
image = image.resize((num_px, num_px))  # 调整大小
my_image = np.array(image).reshape((num_px * num_px * 3, 1))  # 确保形状正确，不需要转置

# 预测图像
my_predicted_image = fun.predict(d["w"], d["b"], my_image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
      classes[int(np.squeeze(my_predicted_image))].decode("utf-8") + "\" picture.")
# 显示图像
plt.imshow(image)
plt.axis('off')  # 隐藏坐标轴
plt.show()