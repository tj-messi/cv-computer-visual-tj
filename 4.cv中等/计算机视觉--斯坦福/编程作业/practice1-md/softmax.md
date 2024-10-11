#softmax



##链接google

链接google colab，下拉数据集

	# This mounts your Google Drive to the Colab VM.
	from google.colab import drive
	drive.mount('/content/drive')
	
	# TODO: Enter the foldername in your Drive where you have saved the unzipped
	# assignment folder, e.g. 'cs231n/assignments/assignment1/'
	FOLDERNAME = None
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

##引入库，设置plt

	import random
	import numpy as np
	from cs231n.data_utils import load_CIFAR10
	import matplotlib.pyplot as plt
	
	%matplotlib inline
	plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'
	
	# for auto-reloading extenrnal modules
	# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
	%load_ext autoreload
	%autoreload 2

##分割训练集，测试集，开发集

	def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
	    """
	    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
	    it for the linear classifier. These are the same steps as we used for the
	    SVM, but condensed to a single function.  
	    """
	    # Load the raw CIFAR-10 data
	    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
	    
	    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
	    try:
	       del X_train, y_train
	       del X_test, y_test
	       print('Clear previously loaded data.')
	    except:
	       pass
	
	    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
	    
	    # subsample the data
	    mask = list(range(num_training, num_training + num_validation))
	    X_val = X_train[mask]
	    y_val = y_train[mask]
	    mask = list(range(num_training))
	    X_train = X_train[mask]
	    y_train = y_train[mask]
	    mask = list(range(num_test))
	    X_test = X_test[mask]
	    y_test = y_test[mask]
	    mask = np.random.choice(num_training, num_dev, replace=False)
	    X_dev = X_train[mask]
	    y_dev = y_train[mask]
	    
	    # Preprocessing: reshape the image data into rows
	    X_train = np.reshape(X_train, (X_train.shape[0], -1))
	    X_val = np.reshape(X_val, (X_val.shape[0], -1))
	    X_test = np.reshape(X_test, (X_test.shape[0], -1))
	    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
	    
	    # Normalize the data: subtract the mean image
	    mean_image = np.mean(X_train, axis = 0)
	    X_train -= mean_image
	    X_val -= mean_image
	    X_test -= mean_image
	    X_dev -= mean_image
	    
	    # add bias dimension and transform into columns
	    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
	    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
	    
	    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev
	
	
	# Invoke the above function to get our data.
	X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
	print('Train data shape: ', X_train.shape)
	print('Train labels shape: ', y_train.shape)
	print('Validation data shape: ', X_val.shape)
	print('Validation labels shape: ', y_val.shape)
	print('Test data shape: ', X_test.shape)
	print('Test labels shape: ', y_test.shape)
	print('dev data shape: ', X_dev.shape)
	print('dev labels shape: ', y_dev.shape)

结果
	
	Train data shape:  (49000, 3073)
	Train labels shape:  (49000,)
	Validation data shape:  (1000, 3073)
	Validation labels shape:  (1000,)
	Test data shape:  (1000, 3073)
	Test labels shape:  (1000,)
	dev data shape:  (500, 3073)
	dev labels shape:  (500,)

##softmax实现

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1728621846091.png)

	def softmax_loss_naive(W, X, y, reg):
	    """
	    Softmax loss function, naive implementation (with loops)
	
	    Inputs have dimension D, there are C classes, and we operate on minibatches
	    of N examples.
	
	    Inputs:
	    - W: A numpy array of shape (D, C) containing weights.
	    - X: A numpy array of shape (N, D) containing a minibatch of data.
	    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
	      that X[i] has label c, where 0 <= c < C.
	    - reg: (float) regularization strength
	
	    Returns a tuple of:
	    - loss as single float
	    - gradient with respect to weights W; an array of same shape as W
	    """
	    # Initialize the loss and gradient to zero.
	    loss = 0.0
	    dW = np.zeros_like(W)
	
	    #############################################################################
	    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
	    # Store the loss in loss and the gradient in dW. If you are not careful     #
	    # here, it is easy to run into numeric instability. Don't forget the        #
	    # regularization!                                                           #
	    #############################################################################
	    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    #pass
	
	    # 训练集的数量
	    num_train = X.shape[0]
	    # 分类的数量
	    num_classes = W.shape[1]
	    for i in range(num_train):
	      scores = np.dot(X[i],W)
	      scores = np.exp(scores)
	      p=scores/np.sum(scores)
	      loss+= -np.log(p[y[i]])
	
	      for k in range (num_classes):
	        p_k=p[k]
	        if k == y[i]:
	          dW[:,k]+=(p_k-1)*X[i]
	        else:
	          dW[:,k]+=p_k * X[i]
	
	    loss/=num_train
	    dW/=num_train
	    loss+=0.5*reg*np.sum(W*W)
	    dW+=reg*W
	
	    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    return loss, dW

loss: 2.384882
sanity check: 2.302585

##向量化

	def softmax_loss_vectorized(W, X, y, reg):
	    """
	    Softmax loss function, vectorized version.
	
	    Inputs and outputs are the same as softmax_loss_naive.
	    """
	    # Initialize the loss and gradient to zero.
	    loss = 0.0
	    dW = np.zeros_like(W)
	
	    #############################################################################
	    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
	    # Store the loss in loss and the gradient in dW. If you are not careful     #
	    # here, it is easy to run into numeric instability. Don't forget the        #
	    # regularization!                                                           #
	    #############################################################################
	    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    #pass
	    # 训练集的数量
	    num_train = X.shape[0]
	    # 分类的数量
	    num_classes = W.shape[1]
	
	    scores = np.dot(X,W)
	    scores = np.exp(scores)
	
	    p=scores/(np.sum(scores,axis=1,keepdims=True))
	
	    loss+=np.sum(-np.log(p[np.arange(num_train),y]))
	
	    p[np.arange(num_train),y]-=1
	
	    dW = np.dot(X.T,p)
	
	    loss/=num_train
	    loss+=0.5*reg*np.sum(W*W)
	
	    dW/=num_train
	    dW+=reg*W
	
	
	
	    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    return loss, dW

