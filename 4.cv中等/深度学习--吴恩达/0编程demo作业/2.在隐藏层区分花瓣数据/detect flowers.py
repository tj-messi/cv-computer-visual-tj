import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) # 设置指定的随机种子，保证每次运行结果一致

X, Y = load_planar_dataset() # 加载数据集

'''
#可视化数据集:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
plt.show()
'''

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]

'''
print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))
'''

