import os
import sys
import pandas as pd
import csv

def create_csv(fold_head,annotation,csv_name):
    fold_tail = []
    label = []
    # 拿到fold_tail 和 label
    # exp : 'T0/7' +'0' 
    with open(annotation,'r') as f :
        lines = f.readlines()
        fold_tail = [line.split(',')[0]  for line in lines]
        label = [line.split(',')[1][0]  for line in lines]
        
    # 拿counts是数据的长度
    counts = len(fold_tail)
    for i in range(counts) :
        data_fold = os.path.join(fold_head,fold_tail[i])
        frames = [os.path.join(data_fold, file) for file in os.listdir(data_fold) if os.path.isfile(os.path.join(data_fold, file))]
        # print(len(frames))

        data_list = [data_fold, str(len(frames)), str(label[i])]
        print(data_list)
   
        with open(csv_name, mode='a', newline='') as file:  
            writer = csv.writer(file, delimiter=' ')
            writer.writerow(data_list) 


if __name__ == '__main__':
    create_csv("/media/tongji/VideoMAEv2-master/data/USVideo_final/val",
               "/media/tongji/VideoMAEv2-master/data/USVideo_final/annotation/val.txt",
               "/media/tongji/VideoMAEv2-master/data/US_annotation/val.csv")