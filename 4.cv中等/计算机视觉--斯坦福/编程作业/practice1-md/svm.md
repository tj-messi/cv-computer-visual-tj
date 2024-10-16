#svm（向量支持机）

##设置google colab

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

申请访问google云盘并且下载对应数据集

##引入库，设置matplotlib

	# Run some setup code for this notebook.
	import random
	import numpy as np
	from cs231n.data_utils import load_CIFAR10
	import matplotlib.pyplot as plt
	
	# This is a bit of magic to make matplotlib figures appear inline in the
	# notebook rather than in a new window.
	%matplotlib inline
	plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'
	
	# Some more magic so that the notebook will reload external python modules;
	# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
	%load_ext autoreload
	%autoreload 2

##处理数据集

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

结果：

	Training data shape:  (50000, 32, 32, 3)
	Training labels shape:  (50000,)
	Test data shape:  (10000, 32, 32, 3)
	Test labels shape:  (10000,)

##展示部分数据集

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

##切割数据

	# Split the data into train, val, and test sets. In addition we will
	# create a small development set as a subset of the training data;
	# we can use this for development so our code runs faster.
	num_training = 49000
	num_validation = 1000
	num_test = 1000
	num_dev = 500
	
	# Our validation set will be num_validation points from the original
	# training set.
	mask = range(num_training, num_training + num_validation)
	X_val = X_train[mask]
	y_val = y_train[mask]
	
	# Our training set will be the first num_train points from the original
	# training set.
	mask = range(num_training)
	X_train = X_train[mask]
	y_train = y_train[mask]
	
	# We will also make a development set, which is a small subset of
	# the training set.
	mask = np.random.choice(num_training, num_dev, replace=False)
	X_dev = X_train[mask]
	y_dev = y_train[mask]
	
	# We use the first num_test points of the original test set as our
	# test set.
	mask = range(num_test)
	X_test = X_test[mask]
	y_test = y_test[mask]
	
	print('Train data shape: ', X_train.shape)
	print('Train labels shape: ', y_train.shape)
	print('Validation data shape: ', X_val.shape)
	print('Validation labels shape: ', y_val.shape)
	print('Test data shape: ', X_test.shape)
	print('Test labels shape: ', y_test.shape)

结果：

	Train data shape:  (49000, 32, 32, 3)
	Train labels shape:  (49000,)
	Validation data shape:  (1000, 32, 32, 3)
	Validation labels shape:  (1000,)
	Test data shape:  (1000, 32, 32, 3)
	Test labels shape:  (1000,)

##reshape数据

	# Preprocessing: reshape the image data into rows
	X_train = np.reshape(X_train, (X_train.shape[0], -1))
	X_val = np.reshape(X_val, (X_val.shape[0], -1))
	X_test = np.reshape(X_test, (X_test.shape[0], -1))
	X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
	
	# As a sanity check, print out the shapes of the data
	print('Training data shape: ', X_train.shape)
	print('Validation data shape: ', X_val.shape)
	print('Test data shape: ', X_test.shape)
	print('dev data shape: ', X_dev.shape)

结果：
	
	Training data shape:  (49000, 3072)
	Validation data shape:  (1000, 3072)
	Test data shape:  (1000, 3072)
	dev data shape:  (500, 3072)

##预处理数据

	# Preprocessing: subtract the mean image
	# first: compute the image mean based on the training data
	mean_image = np.mean(X_train, axis=0)
	print(mean_image[:10]) # print a few of the elements
	plt.figure(figsize=(4,4))
	plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
	plt.show()
	
	# second: subtract the mean image from train and test data
	#也就是执行归一化
	X_train -= mean_image
	X_val -= mean_image
	X_test -= mean_image
	X_dev -= mean_image
	
	# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
	# only has to worry about optimizing a single weight matrix W.
	#在每个数据集的末尾添加了一个全为1的列。这是为了实现所谓的“偏置技巧”（bias trick），在机器学习模型中，偏置项通常用于调整模型的输出，使其能够更好地拟合数据。
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
	X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1728468640496.png)

##SVM

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241009181641.png)

其中

Wj * Xi 是错误分类的分数

-Wyi * Xi 是正确分类的分数

所以一旦分类错误，就可以计算出对于该损失函数的梯度

其中正确分类的Wj所在的梯度是 -Xi

错误分类的Wyi所在的梯度是 +Xi

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/468dda25d51c0a367b3759a6fe205d1.jpg)

	def svm_loss_naive(W, X, y, reg):
	    """
	    Structured SVM loss function, naive implementation (with loops).
	
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
	    dW = np.zeros(W.shape)  # initialize the gradient as zero
	
	    # compute the loss and the gradient
	    num_classes = W.shape[1]
	    num_train = X.shape[0]
	    loss = 0.0
	    for i in range(num_train):
	        scores = X[i].dot(W)
	        correct_class_score = scores[y[i]]
	        for j in range(num_classes):
	            if j == y[i]:
	                continue
	            margin = scores[j] - correct_class_score + 1  # note delta = 1
	            if margin > 0:
	                loss += margin
	                # 正确分类的梯度减上X[i]
	                dW[:,y[i]] -= X[i].T
	                # 错误分类的梯度加去X[i]
	                dW[:,j] += X[i].T
	
	    # Right now the loss is a sum over all training examples, but we want it
	    # to be an average instead so we divide by num_train.
	    loss /= num_train
	
	    # Add regularization to the loss.
	    loss += reg * np.sum(W * W)
	
	    #############################################################################
	    # TODO:                                                                     #
	    # Compute the gradient of the loss function and store it dW.                #
	    # Rather that first computing the loss and then computing the derivative,   #
	    # it may be simpler to compute the derivative at the same time that the     #
	    # loss is being computed. As a result you may need to modify some of the    #
	    # code above to compute the gradient.                                       #
	    #############################################################################
	    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    # 梯度同样处理
	    dW /= num_train
	    # 正则项的梯度
	    dW += 2 * reg * W
	
	    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    return loss, dW

##向量化

	from builtins import range
	import numpy as np
	from random import shuffle
	from past.builtins import xrange
	
	
	def svm_loss_naive(W, X, y, reg):
	    """
	    Structured SVM loss function, naive implementation (with loops).
	
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
	    dW = np.zeros(W.shape)  # initialize the gradient as zero
	
	    # compute the loss and the gradient
	    num_classes = W.shape[1]
	    num_train = X.shape[0]
	    loss = 0.0
	    for i in range(num_train):
	        scores = X[i].dot(W)
	        correct_class_score = scores[y[i]]
	        for j in range(num_classes):
	            if j == y[i]:
	                continue
	            margin = scores[j] - correct_class_score + 1  # note delta = 1
	            if margin > 0:
	                loss += margin
	                # 正确分类的梯度减上X[i]
	                dW[:,y[i]] -= X[i].T
	                # 错误分类的梯度加去X[i]
	                dW[:,j] += X[i].T
	
	    # Right now the loss is a sum over all training examples, but we want it
	    # to be an average instead so we divide by num_train.
	    loss /= num_train
	
	    # Add regularization to the loss.
	    loss += reg * np.sum(W * W)
	
	    #############################################################################
	    # TODO:                                                                     #
	    # Compute the gradient of the loss function and store it dW.                #
	    # Rather that first computing the loss and then computing the derivative,   #
	    # it may be simpler to compute the derivative at the same time that the     #
	    # loss is being computed. As a result you may need to modify some of the    #
	    # code above to compute the gradient.                                       #
	    #############################################################################
	    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    # 梯度同样处理
	    dW /= num_train
	    # 正则项的梯度
	    dW += 2 * reg * W
	
	    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    return loss, dW
	
	
	
	def svm_loss_vectorized(W, X, y, reg):
	    """
	    Structured SVM loss function, vectorized implementation.
	
	    Inputs and outputs are the same as svm_loss_naive.
	    """
	    loss = 0.0
	    dW = np.zeros(W.shape)  # initialize the gradient as zero
	
	    #############################################################################
	    # TODO:                                                                     #
	    # Implement a vectorized version of the structured SVM loss, storing the    #
	    # result in loss.                                                           #
	    #############################################################################
	    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    num_classes = W.shape[1]
	    num_train = X.shape[0]
	    scores = np.dot(X,W)
	    scores_correct = scores[range(num_train),y].reshape((scores.shape[0],1))
	    margins = np.maximum(0,scores - scores_correct + 1)
	    margins[range(num_train),y] = 0
	    loss += np.sum(margins) / num_train
	    loss += reg * np.sum(W * W)
	    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    #############################################################################
	    # TODO:                                                                     #
	    # Implement a vectorized version of the gradient for the structured SVM     #
	    # loss, storing the result in dW.                                           #
	    #                                                                           #
	    # Hint: Instead of computing the gradient from scratch, it may be easier    #
	    # to reuse some of the intermediate values that you used to compute the     #
	    # loss.                                                                     #
	    #############################################################################
	    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    margins[margins > 0] = 1
	    row_sum = np.sum(margins,axis = 1)
	    margins[range(num_train),y] = -row_sum
	    dW += np.dot(X.T, margins)/num_train + reg * W 
	    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	    return loss, dW

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1728481023961.png)

##线性分类训练

	class LinearClassifier(object):
	    def __init__(self):
	        self.W = None
	
	    def train(
	        self,
	        X,
	        y,
	        learning_rate=1e-3,
	        reg=1e-5,
	        num_iters=100,
	        batch_size=200,
	        verbose=False,
	    ):
	        """
	        Train this linear classifier using stochastic gradient descent.
	
	        Inputs:
	        - X: A numpy array of shape (N, D) containing training data; there are N
	          training samples each of dimension D.
	        - y: A numpy array of shape (N,) containing training labels; y[i] = c
	          means that X[i] has label 0 <= c < C for C classes.
	        - learning_rate: (float) learning rate for optimization.
	        - reg: (float) regularization strength.
	        - num_iters: (integer) number of steps to take when optimizing
	        - batch_size: (integer) number of training examples to use at each step.
	        - verbose: (boolean) If true, print progress during optimization.
	
	        Outputs:
	        A list containing the value of the loss function at each training iteration.
	        """
	        num_train, dim = X.shape
	        num_classes = (
	            np.max(y) + 1
	        )  # assume y takes values 0...K-1 where K is number of classes
	        if self.W is None:
	            # lazily initialize W
	            self.W = 0.001 * np.random.randn(dim, num_classes)
	
	        # Run stochastic gradient descent to optimize W
	        loss_history = []
	        for it in range(num_iters):
	            X_batch = None
	            y_batch = None
	
	            #########################################################################
	            # TODO:                                                                 #
	            # Sample batch_size elements from the training data and their           #
	            # corresponding labels to use in this round of gradient descent.        #
	            # Store the data in X_batch and their corresponding labels in           #
	            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
	            # and y_batch should have shape (batch_size,)                           #
	            #                                                                       #
	            # Hint: Use np.random.choice to generate indices. Sampling with         #
	            # replacement is faster than sampling without replacement.              #
	            #########################################################################
	            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	            choice_idxs = np.random.choice(num_train,batch_size)
	            X_batch = X[choice_idxs]
	            y_batch = y[choice_idxs]
	
	            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	            # evaluate loss and gradient
	            loss, grad = self.loss(X_batch, y_batch, reg)
	            loss_history.append(loss)
	
	            # perform parameter update
	            #########################################################################
	            # TODO:                                                                 #
	            # Update the weights using the gradient and the learning rate.          #
	            #########################################################################
	            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	            
	            self.W -= learning_rate * grad
	
	            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	            if verbose and it % 100 == 0:
	                print("iteration %d / %d: loss %f" % (it, num_iters, loss))
	
	        return loss_history

调用训练

	# In the file linear_classifier.py, implement SGD in the function
	# LinearClassifier.train() and then run it with the code below.
	from cs231n.classifiers import LinearSVM
	svm = LinearSVM()
	tic = time.time()
	loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
	                      num_iters=1500, verbose=True)
	toc = time.time()
	print('That took %fs' % (toc - tic))

损失

	iteration 0 / 1500: loss 796.876175
	iteration 100 / 1500: loss 472.439065
	iteration 200 / 1500: loss 286.561045
	iteration 300 / 1500: loss 175.174452
	iteration 400 / 1500: loss 107.294658
	iteration 500 / 1500: loss 66.577661
	iteration 600 / 1500: loss 42.454063
	iteration 700 / 1500: loss 27.529834
	iteration 800 / 1500: loss 18.724411
	iteration 900 / 1500: loss 13.755224
	iteration 1000 / 1500: loss 10.751597
	iteration 1100 / 1500: loss 8.385586
	iteration 1200 / 1500: loss 7.456650
	iteration 1300 / 1500: loss 6.583702
	iteration 1400 / 1500: loss 5.222297
	That took 10.588674s

展示图

	plt.plot(loss_hist)
	plt.xlabel('Iteration number')
	plt.ylabel('Loss value')
	plt.show()

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241009215819.png)

##预测
	
	def predict(self, X):
	        """
	        Use the trained weights of this linear classifier to predict labels for
	        data points.
	
	        Inputs:
	        - X: A numpy array of shape (N, D) containing training data; there are N
	          training samples each of dimension D.
	
	        Returns:
	        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
	          array of length N, and each element is an integer giving the predicted
	          class.
	        """
	        y_pred = np.zeros(X.shape[0])
	        ###########################################################################
	        # TODO:                                                                   #
	        # Implement this method. Store the predicted labels in y_pred.            #
	        ###########################################################################
	        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
	        #pass
	        scores=np.dot(X,self.W)
	        y_pred = np.argmax(scores,axis=1)
	
	        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	        return y_pred

预测

	training accuracy: 0.379571
	validation accuracy: 0.385000