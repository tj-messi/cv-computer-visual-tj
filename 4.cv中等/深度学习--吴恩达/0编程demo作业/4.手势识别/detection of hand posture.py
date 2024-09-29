import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

#%matplotlib inline
np.random.seed(1)

# 定义初始参数 y_hat 和 y
y_hat = tf.Variable(36, dtype=tf.float32, name='y_hat')  # y_hat变量，初始值为36
y = tf.constant(39, dtype=tf.float32, name='y')            # y常量，设置为39

# 定义学习率
learning_rate = 0.01

# 创建优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

for step in range(100):  # 进行100次迭代
    with tf.GradientTape() as tape:
        # 计算损失
        loss = (y - y_hat) ** 2  # 计算损失

    # 计算梯度
    gradients = tape.gradient(loss, [y_hat])

    # 更新 y_hat
    optimizer.apply_gradients(zip(gradients, [y_hat]))

    # 打印每10步的损失和 y_hat 值
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.numpy()}, y_hat: {y_hat.numpy()}")

# 最终输出
print(f"Final Loss: {loss.numpy()}, Final y_hat: {y_hat.numpy()}")