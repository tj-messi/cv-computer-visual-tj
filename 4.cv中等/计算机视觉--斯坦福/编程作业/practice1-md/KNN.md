#K近邻分类

##设置google.colab

	# This mounts your Google Drive to the Colab VM.
	from google.colab import drive
	drive.mount('/content/drive')

	# TODO: Enter the foldername in your Drive where you have saved the unzipped
	# assignment folder, e.g. 'cs231n/assignments/assignment1/'
	FOLDERNAME = 'cs231n/assignments/practice1/'
	assert FOLDERNAME is not None, "[!] Enter the foldername."

	# Now that we've mounted your Drive, this ensures that
	# the Python interpreter of the Colab VM can load
	# python files from within it.
	import sys
	sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

	# This downloads the CIFAR-10 dataset to your Drive
	# if it doesn't already exist.
	%cd /content/drive/My\ Drive/$FOLDERNAME/cs231n/datasets/
	!bash get_datasets.sh
	%cd /content/drive/My\ Drive/$FOLDERNAME

会申请访问google云盘并且下载好需要的数据集

##拉入数据集

	# Run some setup code for this notebook.

	import random
	import numpy as np
	from cs231n.data_utils import load_CIFAR10
	import matplotlib.pyplot as plt

	# This is a bit of magic to make matplotlib figures appear inline in the notebook
	# rather than in a new window.
	%matplotlib inline
	plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots 设置plt图像大小
	plt.rcParams['image.interpolation'] = 'nearest' #设置plt图像插入形式
	plt.rcParams['image.cmap'] = 'gray' #设置图像为灰度图

	# Some more magic so that the notebook will reload external python modules;
	# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
	%load_ext autoreload
	%autoreload 2

打印数据集格式

	# Load the raw CIFAR-10 data.
	cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
	
	# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
	try:
	   del X_train, y_train
	   del X_test, y_test
	   print('Clear previously loaded data.')
	except:
	   pass
	
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
	
	# As a sanity check, we print out the size of the training and test data.
	print('Training data shape: ', X_train.shape)
	print('Training labels shape: ', y_train.shape)
	print('Test data shape: ', X_test.shape)
	print('Test labels shape: ', y_test.shape)

数据集的格式大小：
	
	Training data shape:  (50000, 32, 32, 3)
	
	Training labels shape:  (50000,)
	
	Test data shape:  (10000, 32, 32, 3)
	
	Test labels shape:  (10000,)

##可视化数据集

先定义一个类的集合。然后取出集合长度。设定一个准备取出的集合个数。这个enumerate(classes)就是枚举同时取出y索引和cls类别的遍历。

然后遍历classes，np.flatnonzero(y_train == y) 返回所有属于当前类别 y 的训练样本的索引。

np.random.choice(idxs, samples_per_class, replace=False) 从这些索引中随机选择 samples_per_class 个不重复的样本索引（集合）

然后同样的方法遍历idxs，计算它在绘图网格中的位置 plt_idx。这里使用了一个复杂的索引计算方法，以便在多个类别和样本间正确排列图像

plt.subplot(samples_per_class, num_classes, plt_idx) 创建一个子图。

plt.imshow(X_train[idx].astype('uint8')) 在子图中显示图像。这里将图像数据转换为无符号8位整数格式，适合显示。

plt.axis('off') 关闭坐标轴。

如果是当前类别的第一个样本（i == 0），则在图像上方显示类别名称作为标题。

	# Visualize some examples from the dataset.
	# We show a few examples of training images from each class.
	classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	num_classes = len(classes)
	samples_per_class = 7
	for y, cls in enumerate(classes):
	    idxs = np.flatnonzero(y_train == y)
	    idxs = np.random.choice(idxs, samples_per_class, replace=False)
	    for i, idx in enumerate(idxs):
	        plt_idx = i * num_classes + y + 1
	        plt.subplot(samples_per_class, num_classes, plt_idx)
	        plt.imshow(X_train[idx].astype('uint8'))
	        plt.axis('off')
	        if i == 0:
	            plt.title(cls)
	plt.show()

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1728441859553.png)

##压缩数据量
	
	num_training = 5000
	mask = list(range(num_training))
	X_train = X_train[mask]
	y_train = y_train[mask]
	
	num_test = 500
	mask = list(range(num_test))
	X_test = X_test[mask]
	y_test = y_test[mask]
	
	# Reshape the image data into rows
	X_train = np.reshape(X_train, (X_train.shape[0], -1))
	X_test = np.reshape(X_test, (X_test.shape[0], -1))
	print(X_train.shape, X_test.shape)

##引入K邻近

	from cs231n.classifiers import KNearestNeighbor
	
	# Create a kNN classifier instance. 
	# Remember that training a kNN classifier is a noop: 
	# the Classifier simply remembers the data and does no further processing 
	classifier = KNearestNeighbor()
	classifier.train(X_train, y_train)

##实现L2距离计算
	
	# Open cs231n/classifiers/k_nearest_neighbor.py and implement
	# compute_distances_two_loops.
	
	# Test your implementation:
	dists = classifier.compute_distances_two_loops(X_test)
	print(dists.shape)

L2：

	dists[i][j]=np.sqrt(np.sum(np.square(X[i]-self.X_train[j])))


#输出前k预测值


	在给出的代码片段中，我们看到了一行使用 NumPy 的 np.argsort 函数的代码。这行代码的目的是从一个训练数据集 self.y_train 中找出与某个给定距离数组 dists[i] 最接近的 k 个点的目标值（即标签）。让我们逐步解释这一行代码：
	
	变量解释：
	self.y_train：这是一个数组，包含了训练数据集的目标值（标签）。
	dists[i]：这是一个数组，包含了某个点（可能是测试集中的点）到训练集中所有点的距离。i 在这里是一个索引，指定了我们正在考虑哪个点的距离。
	k：这是一个整数，表示我们想要找出的最近邻的个数。
	np.argsort 的作用：
	np.argsort(dists[i])：这个函数会对 dists[i] 数组中的元素进行排序，并返回排序后的索引数组。也就是说，如果 dists[i] 中的第 3 个元素是最小的，那么 np.argsort(dists[i]) 的第一个元素就会是 3。
	切片操作：
	[:k]：这个切片操作会从排序后的索引数组中取出前 k 个索引。这些索引对应了 dists[i] 中最小的 k 个元素。
	索引self.y_train：
	使用从 np.argsort 返回的索引数组（经过切片操作后只保留了前 k 个）来索引 self.y_train 数组。这样，我们就能得到与给定点距离最近的 k 个训练点的目标值。

拿出前k的值：

	closest_y=self.y_train[np.argsort(dists[i][:k])]

计算出现最多的标签：

	y_pred[i] = np.argmax(np.bincount(closest_y))

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1728453960891.png)