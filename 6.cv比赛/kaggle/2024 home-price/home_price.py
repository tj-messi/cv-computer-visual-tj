import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.utils.data

# 输出torch版本号
# print(torch.__version__)
# 设置pytorch中默认的浮点类型
torch.set_default_dtype(torch.float32)  # 使用torch.float32替代torch.FloatTensor

# 使用pandas读取两个文件
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')


# 训练数据集包括1460个样本、80个特征和1个标签
# print(train_data.shape)
# 测试数据集包括1459个样本和80个特征，需要将测试数据集中每个样本的标签预测出来
# print(test_data.shape)

# [:,1:-1]所有行的第一列到-2列，[:,1:]
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))


#print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 打印所有特征的数量 (2919,79)
# print(all_features.shape)



# object是str类或数字混合类型（mixed），将特征为数字的列单独拿出。并保存列名到numeric
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  # 排除object类型的特征

# 打印非离散的数据类型
print(numeric_features)

# 对每种数字型的特征(每一的数字)进行标准化
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

# 填充缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# dummpy_na=True将缺失值也当做合法的特征值并为其创建指示特征
# get_dummies即上述将离散值转换为指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)

# 再次检查数据类型
# print(all_features.dtypes)

print(all_features.shape)
#print(all_features)

# 取train_data的行数，即训练集个数
n_train = train_data.shape[0] 

# 训练集特征
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
# 测试集特征
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
# 训练集标签
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

## 模型训练
loss = torch.nn.MSELoss()

def get_net(feature_num):
	# 实例化nn
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()

def train(net,train_features,train_labels,test_features,test_labels,
          num_epochs,learning_rate,weight_decay,batch_size):
    # 训练损失，测试损失
    train_ls, test_ls = [], []
    # 加载数据
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

     # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()

    # 开始训练
    for num_epochs in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())

            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            l.backward()
            # 更新参数
            optimizer.step()
        # 记录训练损失
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))

    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    # k需要大于1,assert用于判断一个表达式，在表达式条件为false的时候出发异常。
    assert k > 1
    # 划分成k个集合，fold_size每个集合的个数
    fold_size = X.shape[0] // k
    # 初始化X_train, y_train
    X_train, y_train = None, None
    # 循环访问k个集合
    for j in range(k):
    	# slice(start, stop[, step])；start:开始位置，stop:结束位置，step：间距
		# 则idx等于就是第j个集合切片对象的集合
        idx = slice(j * fold_size, (j + 1) * fold_size)
        # 将第j个集合的特征，和第j个集合的标签分别放在X_part, y_part
        X_part, y_part = X[idx, :], y[idx]
        # 如果当前的集合是第i折交叉验证，就将当前的集合当作验证模型
        # （j=i）就是当前所取部分刚好是要当验证集的部分
        if j == i:
            # 将当前所取集合放进验证集
            X_valid, y_valid = X_part, y_part
        # 如果j!=i且X_train是空的则直接将此部分放进训练集
        elif X_train is None:
            X_train, y_train = X_part, y_part
        # 如果j!=i且访问到其余除了验证集（j=i）其余集合的子集，就使用concat连接已经放进训练集的集合
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    # 依次返回训练模型和第i个验证模型
    return X_train, y_train, X_valid, y_valid

import matplotlib.pyplot as plt

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """
    使用对数y轴绘制图表的函数

    参数:
    x_vals: x轴的值
    y_vals: y轴的值
    x_label: x轴的标签
    y_label: y轴的标签
    x2_vals: 第二个x轴的值（可选）
    y2_vals: 第二个y轴的值（可选）
    legend: 图例（可选）
    figsize: 图形大小（可选）

    返回:
    None
    """
    plt.figure(figsize=figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
    if legend:
        plt.legend(legend)
    plt.show()

def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    # 初始化训练集损和验证集的损失和
    train_l_sum, valid_l_sum = 0, 0
    # 依次访问所划分集合
    for i in range(k):
    	# 获得训练模型和第i个验证模型
        data = get_k_fold_data(k, i, X_train, y_train)
        # 获得net实例
        net = get_net(X_train.shape[1])
        # 分别计算出训练集与验证集损失
        # *data可以分别读取数据，在这里等于是得到了get_k_fold_data)函数的return的四个表，直接写data会导致缺位置参数。
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        #计算训练集损失之和                     
        train_l_sum += train_ls[-1]
        #计算验证集损失之和
        valid_l_sum += valid_ls[-1]
        # 画图
        if i == 0:
        	# d21:自己封装的包，包里是常用的函数模块
            semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 6, 100, 6, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    # 获得net实例
    net = get_net(train_features.shape[1])
    # 返回训练集损失
    # 单下划线 _ 单个独立下划线是用作一个名字，来表示某个变量是临时的或无关紧要的。
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    # 作图
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    # 计算预测标签
    # detach:返回一个新的tensor,从当前图形分离（官方文档解释）
    preds = net(test_features).detach().numpy()
    # SalePrice标签列
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    # 将测试集的Id和预测结果拼接在一起，axis沿水平方向拼接，与上面默认纵向拼接不同。
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    # 转成可提交的csv格式
    submission.to_csv('./submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)


