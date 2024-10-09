import numpy as np
import random
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(10.0,8.0)#修改图片大小
plt.rcParams['image.interpolation']='nearest'#修改插值方式
plt.rcParams['image.cmap']='gray'#使用灰度图

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)