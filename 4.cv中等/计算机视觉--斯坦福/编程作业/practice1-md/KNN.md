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
