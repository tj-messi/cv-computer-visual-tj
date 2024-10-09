#numpy

先定义

	import numpy as np

##A
###numpy.argsort
用于返回数组排序后的索引。这些索引是基于数组元素的大小进行排序后得到的，通常用于需要知道元素在排序后数组中的位置时
	
	x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])  
	  
	# 获取排序后的索引  
	sorted_indices = np.argsort(x)  
	  
	print(sorted_indices)
	输出将是：
	
	[ 1  3  6  0  2  9 10  4  8  7]

##F
###np.flatnonzero
返回所有输入bool值为非0的内容，接受一个数组，返回索引值数组

	np.flatnonzero(bool)

	#示例

	a = np.array([[1,2,3,4,3],
	              [3,4,5,3,1]])
	idx = np.flatnonzero(a==3)
	print(idx)
	#示例输出
	[2 4 5 8]

##R
###np.random.choice
这是 NumPy 库中的一个函数，用于从给定的一维数组中随机抽取元素

**idxs**：这是你要从中抽取元素的一维数组。在这个上下文中，idxs 通常包含之前通过某种条件筛选出来的索引。

**samples_per_class**：这是一个整数，表示你想要从 idxs 中随机抽取的元素数量。在这个上下文中，它通常表示每个类别你想要抽取的样本数量。

**replace=False**：这是一个布尔参数，表示在抽取元素时是否允许重复。如果设置为 False，则抽取的元素是不重复的；如果设置为 True，则可能抽取到重复的元素。在这个上下文中，通常设置为 False，以确保每个抽取的索引都是唯一的

	idxs = np.random.choice(idxs, samples_per_class, replace=False)
	#在idxs中随机抽取samples_per_class个内容

###np.reshape
允许你改变一个已有数组的形状（即其维度和大小），而不改变其数据。

	numpy.reshape(a, newshape, order='c')

**a（array_like）**：需要改变形状的数组。

**newshape**（int 或 tuple of ints）：新的形状，通常以一个整数元组的形式给出。例如，(rows, columns)。newshape 中的元素乘积必须等于 a 的元素个数。如果某个维度被指定为 -1，则该维度的大小将被自动计算，以便使得总的元素数量保持不变。

**order（{'C', 'F', 'A', 'K'}，可选）**：这个参数决定了数组在内存中的布局（行优先还是列优先等）。'C' 表示按行优先（C-style，行主序），'F' 表示按列优先（Fortran-style，列主序）。'A' 表示如果数组在内存中是按 Fortran 样式连续的，则以 Fortran 样式进行重塑，否则以 C 样式进行。'K' 表示尽可能保持数组的内存顺序。默认值是 'C'。

##S
###np.sqrt
用于计算数组元素的平方根
	
	# 创建一个数组  
	x = np.array([0, 1, 4, 9, 16])  
	  
	# 计算数组元素的平方根  
	y = np.sqrt(x)	
	
	#输出
	[0. 1. 2. 3. 4.]

###np.square
用于计算数组元素的平方
	
	import numpy as np  
	  
	# 创建一个数组  
	x = np.array([1, 2, 3, 4, 5])  
	  
	# 计算数组元素的平方  
	squared = np.square(x)  
	  
	print(squared)
	
	[ 1  4  9 16 25]

###np.sum
用于计算数组元素的总和
	# 创建一个数组  
	b = np.array([1, 2, 3], dtype=np.float32)  
	
	# 指定返回数据类型为 int64  
	total_with_dtype = np.sum(b, dtype=np.int64)  
	print(total_with_dtype)  # 输出: 6 (类型为 int64)
	
	# 创建一个二维数组  
	y = np.array([[1, 2, 3], [4, 5, 6]])  
	
	# 按列求和  
	column_sum = np.sum(y, axis=0)  
	print(column_sum)  # 输出: [5 7 9]  
	
	# 按行求和  
	row_sum = np.sum(y, axis=1)  
	print(row_sum)  # 输出: [6 15]
	
	# 创建一个数组  
	z = np.array([[1, 2, 3], [4, 5, 6]])  
	
	# 按行求和，并保持维度  
	row_sum_keepdims = np.sum(z, axis=1, keepdims=True)  
	print(row_sum_keepdims)  # 输出: [[ 6]  
	                         #       [15]]