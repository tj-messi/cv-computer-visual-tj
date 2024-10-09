import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(10,8)

# 创建一些数据  
x = [1, 2, 3, 4, 5]  
y = [1, 4, 9, 16, 25]  
  
# 使用数据创建一个图表  
plt.plot(x, y)  
  
# 显示图表  
plt.show()