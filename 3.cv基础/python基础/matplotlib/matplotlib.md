#matplotlib

先定义 

	import matplotlib.pyplot as plt

##A

##I

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

