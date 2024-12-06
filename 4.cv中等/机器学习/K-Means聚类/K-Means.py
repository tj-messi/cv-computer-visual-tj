import csv
import sys
import numpy as np
from math import sqrt
import random
from matplotlib import pyplot as plt

def load_dataset(filename,seperator):
    """数据的导入"""
    data_list=[]
    fr=open(filename)
    for line in fr.readlines():
        curline=line.strip().split(seperator)
        float_list=[float(element) for element in curline if element]
        data_list.append(float_list)
    return data_list

def show_cluster(dataset,central_points,result,save_name):
    """可视化：散点图的显示以及保存"""
    points_num=len(central_points)
    data_num=len(result)
    color = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "gray", "brown", "cyan", "magenta", "gold"]
    for i in range(data_num):
        data_color=color[int(result[i][0])]
        plt.plot(dataset[i][0],dataset[i][1],marker='o',color=data_color)
    for j in range(points_num):
        plt.plot(central_points[j][0], central_points[j][1],marker='^',color='k')
    plt.savefig(str(save_name))
    plt.show()

def showElbow(dataset,max_centers):
    """计算手肘法对应的折线图"""
    k_distance=[]
    for k in range(max_centers):
        sum_all=0
        for i in range(1):      #每一个K进行50次，最后k_distance取其平均值
            central_points, result = Kmeans(dataset, k)
            sum_all+=sum([sqrt(result[i][1]) for i in range(len(result))])#将每一个欧氏距离误差相加
        k_distance.append(sum_all/10)
    plt.plot(range(1, max_centers + 1), k_distance, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Distances')
    plt.savefig("Elbow")
    plt.show()

def calculateEucl(vector1,vector2):
    """计算两点之间欧式距离"""
    sum=0
    for index in range(len(vector1)):
        sum+=(vector1[index]-vector2[index])**2
    return sqrt(sum)
def make_central_points(dataset,k):
    """构建一个包含k个随机质心的集合：这是k-means的不足之处，留待改进"""
    attributes_num=len(dataset[0])                #特征数量
    central_points=np.zeros((k,attributes_num))   #创建一个k行n列的二维数组
    for index in range(attributes_num):
        index_min=min([child_list[index] for child_list in dataset])
        index_max=max([child_list[index] for child_list in dataset])
        index_range=index_max-index_min
        for child_array in central_points:        #每一个点都是均值点
            child_array[index]=index_min+index_range*np.random.rand()
    return central_points                         #返回一个k行n列的二维数组

def nearset(data,centers):
    """计算该数据点到已经初始化的数据中心的最小距离"""
    num_centers=len(centers)
    distance_list=[calculateEucl(data,centers[i]) for i in range(num_centers)]
    return min(distance_list)

def Kmeanspp_central_points(dataset,k):
    """k-means++初始化中心点的目标是使得中心点的距离尽可能远"""
    data_num=len(dataset)
    attributes_num = len(dataset[0])
    central_points=np.zeros((k,attributes_num))
    random_center_index=np.random.randint(0,data_num)  #随机选择一个样本点作为中心点
    central_points[0]=dataset[random_center_index]         #将这个随机点作为样本点
    distance=[0.0 for i in range(data_num)]                #初始化一个距离的序列
    for i in range(1,k):              #对于每一个中心点
        for j in range(data_num):     #遍历所有的数据点
            distance[j]=nearset(dataset[j],central_points[0:i])    #计算所有数据点最小距离
        sum_all=sum(distance)*random.random()
        for index in range(len(distance)):
            sum_all-=distance[index]
            if sum_all<0:
                central_points[i]=dataset[index]
                break
    return central_points

def Kmeans(dataset,k):
    """K-means聚类算法"""
    data_num=len(dataset)
    if data_num==0:
        return [[]],[[]]
    attributes_num = len(dataset[0])
    result=np.zeros((data_num,2))                 #创建一个data_num行2列的二维数组,保存每一个数据点的信息
    central_points=Kmeanspp_central_points(dataset,k)
    loop=True
    while loop:
        loop=False
        #形成以每一个中心点为核心的类簇
        for data_index in range(data_num):         #计算每一个数据点到中心点的距离
            min_dis,min_index=float('inf'),-1
            for points_index in range(k):          #求该数据点到这些中心点的最小距离
                distance=calculateEucl(dataset[data_index],central_points[points_index])
                if distance<min_dis:
                    min_dis,min_index=distance,points_index
            result[data_index][1]=min_dis**2       #修改每一次的最小距离
            if result[data_index][0]!=min_index:   #形成类簇
                result[data_index][0],loop=min_index,True
        for points_index in range(k):  #遍历每一个中心点,修改中心点
            #含有所有子元素的列表
            child_datalist=[dataset[i] for i in range(data_num) if result[i][0]==points_index]
            if len(child_datalist)==0:
                continue
            for attribute in range(attributes_num):
                # 含有每一个子元素的相应属性的列表
                attribute_list=[child[attribute] for child in child_datalist]
                #修改中心点的坐标
                central_points[points_index][attribute]=sum(attribute_list)/len(attribute_list)
    return central_points,result

def BiKmeans(dataset,k,judge):
    """二分Kmeans算法"""
    data_num = len(dataset)
    record=np.zeros((data_num,2))                      #保存每一个数据点的中心点以及欧氏距离
    central_points=make_central_points(dataset,1)   #质心初始化为所有质点的平均值,[[0]]
    central_points=central_points.tolist()
    for j in range(data_num):                          #先计算每一个点到该中心点的距离
        record[j][1]=calculateEucl(central_points[0],dataset[j])**2   #计算每一个数据点初始的欧氏距离
    while len(central_points)<k:                       #在达到目标个数据点之前都做循环
        if judge:
            show_cluster(dataset, central_points, record,len(central_points))
            print("程序仍在进行(分步可视化)，请稍等")
        min_dis=float('inf')
        choose_point_index=-1             #表示选择哪一个中心点进行二分
        add_centers,data_result=[[]],[[]] #二分后的中心坐标、该簇下二分后的result
        for i in range(len(central_points)):           #对于中心点列表中的每一个点
            #获取第i个中心点下的所有数据点
            child_data_list=[dataset[index] for index in range(data_num) if record[index][0]==i]
            if len(child_data_list)==0:    #如果该中心点没有子点，则一定不进行二分
                continue
            temp_centers,temp_result=Kmeans(child_data_list,2)
            split_sse=sum([temp_result[i][1] for i in range(len(temp_result))])   #计算所有点到各自中心点距离的平方和
            splitnot_sse=sum([record[index][1] for index in range(data_num) if record[index][0]!=i])
            if split_sse+splitnot_sse<min_dis:
                choose_point_index,min_dis,add_centers=i,split_sse+splitnot_sse,temp_centers
                data_result=temp_result.copy()         #对第i号簇进行的划分后的点集
        #改变data_result的索引
        for i in range(len(data_result)):
            if int(data_result[i][0])==0:       #用第0号分点替换choose_point_index
                data_result[i][0]=choose_point_index
            elif int(data_result[i][0])==1:       #将第1号分点加在central_points最后
                data_result[i][0]= len(central_points)
        #更新中心点
        central_points[choose_point_index]=add_centers[0]    #将已经被分割的第i号簇变为第0号子簇
        central_points.append(add_centers[1])                #再加入第1号子簇
        #更新record
        j=0
        for i in range(data_num):
            if record[i][0]==choose_point_index:              #该数据点属于被分割之前的簇
                record[i],j=data_result[j],j+1
    return central_points,record

def main():
    dataset1=load_dataset("data1.csv",',')     #load_dataset()第二个参数为分隔符
    dataset,k=[[]],0
    print("data1,6000条数据")
    dataset=dataset1
    k=3

    

    print("开始Kmeans聚类")
    print("数据集大小为",len(dataset))
    print("目标簇数为",k)
   # print(dataset)

    flag2=input("请选择是否要进行二分Kmeans的分步执行(0(否)或者1(是))")
    judge=0
    if flag2=='0':
        judge=False
    elif flag2=='1':
        judge=True
    else:
        print("输入错误")
        sys.exit(0)
    showElbow(dataset,10)                                                   #是否显示手肘法折线图？
    central_points,result=BiKmeans(dataset,k,judge)                          #二分Kmeans中有分步可视化,更改K的值
    #central_points,result=Kmeans(dataset,3)                                 #正常Kmeans，取点采用了Kmeans++
    show_cluster(dataset,central_points,result,'final')            #保存最后一张结果图名为'final'

if __name__ == '__main__' :
    main()
