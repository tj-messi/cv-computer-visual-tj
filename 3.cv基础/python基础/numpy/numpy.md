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

###np.argmax
用于返回数组中最大值的索引

	numpy.argmax(a, axis=None, out=None)

**a**：输入的数组。

**axis**：可选参数，指定沿哪个轴寻找最大值。如果为 None（默认），则在整个数组中寻找最大值。

**out**：可选参数，指定一个数组来存储结果。如果没有提供，则返回一个新数组。

###np.arrat_split
array_split 是一个用于将数组分割成多个子数组的函数。这个函数非常灵活，可以沿指定的轴将数组分割成指定数量的子数组，或者根据提供的索引列表进行分割。

	print(np.array_split(x, 3))#长度切割
	print(np.array_split(x, [3, 7]))#索引切割 


##B
###np.bincount()
用于计算非负整数数组中每个元素出现的次数。

	numpy.bincount(x, weights=None, minlength=0)

**x**：输入的非负整数数组，其中的每个元素表示一个类别或标签。

**weights**：可选的权重数组，与 x 具有相同的长度。如果提供了权重，np.bincount 将计算加权后的次数。

**minlength**：可选参数，指定输出数组的最小长度。如果 x 中的最大元素大于或等于 minlength，输出数组的长度将是 x 中的最大元素加一；否则，输出数组的长度将是 minlength。

##C
###np.concatenate
用于沿指定的轴连接（拼接）数组序列。这是处理数组时非常常见且有用的操作，特别是在需要将多个数组合并为一个更大的数组时



	# 创建两个二维数组  
	a = np.array([[1, 2, 3], [4, 5, 6]])  
	b = np.array([[7, 8, 9], [10, 11, 12]])  
	  
	# 沿第一个轴连接（垂直连接，增加行数）  
	c_vertical = np.concatenate((a, b), axis=0)  
	print("Vertical concatenation:\n", c_vertical)  
	  
	# 沿第二个轴连接（水平连接，增加列数）  
	c_horizontal = np.concatenate((a, b), axis=1)  
	print("Horizontal concatenation:\n", c_horizontal)

	Vertical concatenation:  
	 [[ 1  2  3]  
	 [ 4  5  6]  
	 [ 7  8  9]  
	 [10 11 12]]  
	  
	Horizontal concatenation:  
	 [[ 1  2  3  7  8  9]  
	 [ 4  5  6 10 11 12]]


###np.compress
用于通过给定的条件（布尔数组）筛选数组元素。这个函数返回一个新数组，仅包含满足条件的元素。

	numpy.compress(condition, a, axis=None, out=None)

**condition**：一个布尔数组，用于指定哪些元素应该被包含在新数组中。condition 的形状必须与 a 的形状相匹配，或者如果指定了 axis，则必须与 a 在该轴上的长度相匹配。

**a**：要筛选的输入数组。

**axis**：沿着它压缩数组的轴。如果为 None，则输入数组会被展平。这是一个可选参数。
##D
###np.dot
矩阵乘法

	np.dot(X, self.X_train.T)

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

##H
###np.hstack
用于沿水平轴（列）堆叠数组序列。该函数将多个数组按照水平方向（即列方向）拼接起来，生成一个新的数组

	import numpy as np  
	  
	# 创建两个数组  
	a = np.array([[1, 2], [3, 4]])  
	b = np.array([[5, 6], [7, 8]])  
	  
	# 水平堆叠  
	result = np.hstack((a, b))  
	  
	print(result)
	
	[[1 2 5 6]  
	 [3 4 7 8]]



##M
###np.mean
用于计算给定数组或数据集的平均值。它可以沿着指定的轴计算均值，或者在没有指定轴的情况下计算整个数组的全局均值

	arr_2d = np.array([[1, 2, 3], [4, 5, 6]])  
	 
	# 沿列计算均值  
	mean_cols = np.mean(arr_2d, axis=0)  
	print(mean_cols)  # 输出: [2.5 3.5 4.5]  
	 
	# 沿行计算均值  
	mean_rows = np.mean(arr_2d, axis=1)  
	print(mean_rows)  # 输出: [2. 5.]

##O
##np.ones
于创建一个数组，并用全1填充这个数组。
	
	numpy.ones(shape, dtype=None, order='C')


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

##T
###np.transpose
用于转置矩阵或数组。转置操作意味着将数组的行转换为列，将列转换为行。
	
	A = np.array([[1, 2], [3, 4]])
	使用 np.transpose(A) 将得到形状为 (n, m) 的数组：
	
	python
	复制代码
	A_transposed = np.transpose(A)  
	print(A_transposed)  
	# 输出:  
	# [[1 3]  
	#  [2 4]]