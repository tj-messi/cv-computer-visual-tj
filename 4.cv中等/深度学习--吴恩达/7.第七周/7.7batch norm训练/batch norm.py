import torch
import torch.nn as nn

bs = 64

print("Pytorch Batch Norm Layer详解")
print("--- 2D input:(mini_batch, num_feature) ---")
# With Learnable Parameters
m = nn.BatchNorm1d(400)  # 例如，房价预测：x的特征数是400，y是房价
# Without Learnable Parameters（无学习参数γ和β）
# m = nn.BatchNorm1d(100, affine=False)
inputs = torch.randn(bs, 400)
print(m(inputs).shape)

print("Batch Norm层的γ和β是要训练学习的参数")
print("γ:", m.state_dict()['weight'].shape)  # gammar
print("β:", m.state_dict()['bias'].shape)  # beta
print("")


print("--- 3D input:(mini_batch, num_feature, other_channel) ---")
m = nn.BatchNorm1d(32)
inputs = torch.randn(bs, 32, 32)  # 这种格式的数据不常用
print(m(inputs).shape)

print("Batch Norm层的γ和β是要训练学习的参数")
print("γ:", m.state_dict()['weight'].shape)  # gammar
print("β:", m.state_dict()['bias'].shape)  # beta
print("")


print("--- 4D input:(mini_batch, num_feature, H, W) ---")
m = nn.BatchNorm2d(3)  # 例如, CIFAR10数据集是三通道的，3x32x32

inputs = torch.randn(bs, 3, 32, 32)
print(m(inputs).shape)

print("Batch Norm层的γ和β是要训练学习的参数")
print("γ:", m.state_dict()['weight'].shape)  # gammar
print("β:", m.state_dict()['bias'].shape)  # beta
print("Batch Norm层的running_mean和running_var是统计量（主要用于预测阶段）")
print("running_mean:", m.state_dict()['running_mean'].shape)
print("running_var:", m.state_dict()['running_var'].shape)

#自己的测试
bn=nn.BatchNorm2d(3)
x=torch.randn([1,3,255,255])
print(x)
y=bn(x)
print(y)

print(bn.running_mean)  # 均值
print(bn.running_var)   # 方差