import numpy as np
import fun
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
#验证x和y的shape是否正确
print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))
'''

'''
# 可以先使用sklrean的LogisticRegressionCV来训练模型，然后画出决策边界，实现一个简单的二分类
# 注意Y的维度不对，需要用ravel()函数将其变成一维数组
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, np.ravel(Y.T))
# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

LR_predictions = clf.predict(X.T)
accuracy = (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100
print('Accuracy of logistic regression: %d %%' % accuracy +
      ' (percentage of correctly labelled datapoints)')

plt.show()
# 准确度在47这样，二分类准确度不高
'''


