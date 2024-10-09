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