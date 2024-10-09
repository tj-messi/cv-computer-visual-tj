#matplotlib

先定义 

	import matplotlib.pyplot as plt

##A
###plt.axis
函数用于控制子图的坐标轴

	plt.axis('off')

'off'：表示关闭坐标轴。这意味着子图将不会显示x轴和y轴
反之on表示打开

##C
###plt.title
用于为当前的子图或图表设置标题。当你想要在图表上方显示一些描述性文本时，这个函数非常有用

	plt.title(label, fontdict=None, loc='center', **kwargs)

**label**：这是一个字符串，表示你想要显示的标题文本。
##I
###plt.imshow

用于在当前的子图中显示一个图像

	X_train[idx].astype('uint8')

X_train[idx].astype('uint8')：这是要显示的图像数据。X_train是一个包含训练图像的数组，idx是当前要显示的图像的索引。.astype('uint8')将图像数据转换为无符号8位整数格式，这是显示图像时常用的格式

##R
###plt.rcRarams

重新设置matplotlib各个属性

重新设置plt图像框的长宽:

	plt.rcParams['figure.figsize']=(high,weight)

重新设置plt图像框的插值的方法：

	plt.rcParams['image.interpolation'] = ''
	#'nearest' 是一种简单的插值方法，它选择最接近目标位置的像素值，不进行任何平滑处理

重新设置图像的颜色映射方式：

	plt.rcParams['image.cmap'] = ''
	#'gray' 是一种灰度颜色映射，它将图像中的像素值直接映射到灰度级别上
	
##S
###plt.subplot

用于在当前的图形窗口中创建一个子图。它允许你在一个大的图形区域内划分出多个小的绘图区域，每个区域都可以独立地显示一个图表或图像

	plt.subplot(samples_per_class, num_classes, plt_idx)

**samples_per_class**：指定子图的行数（即垂直方向上的子图数量）。

**num_classes**：指定子图的列数（即水平方向上的子图数量）。

**plt_idx**：指定当前要创建的子图的索引号，它按照行优先的顺序进行编号。这个编号是按照从左到右从上到下进行

###plt.show
plt.show() 是用于显示图形的命令，它属于 matplotlib 库中的 pyplot 模块。通常，在使用 matplotlib 创建图表或进行数据可视化后，你需要调用 plt.show() 来将图表渲染到屏幕上

	# 创建一些数据  
	x = [1, 2, 3, 4, 5]  
	y = [1, 4, 9, 16, 25]  
	  
	# 使用数据创建一个图表  
	plt.plot(x, y)  
	  
	# 显示图表  
	plt.show()

##X
###plt.xlabel
给图标设置x轴内容

	plt.xlabel('Iteration number')

##Y
###plt.ylabel
给图标设置y轴内容

	plt.ylabel('Loss value')