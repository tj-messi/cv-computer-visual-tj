import torch

# 检查CUDA是否可用
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建张量并移动到GPU上
x = torch.randn(3, 3)
print(x)
x=x*x.T
print(x)

# 创建模型并移动到GPU上
#model = MyModel().to(device)

print(torch.cuda.get_device_name())