[toc]

进入conda虚拟环境

```
conda activate psp_env
```



# 第0步：训练

training的命令，实验进行到后期如果想用其他数据集重新训练模型或者尝试其他新想法，可以使用。

checkpoint_path是预训练模型的位置，可以修改；

数据集的位置在configs/paths_config.py里改，把celeba_train和celeba_test改成要使用的数据集的位置；

exp_dir是存储中间模型checkpoints的位置，如果不想覆盖原来的，也可以改位置；

```
nohup python scripts/train.py \
--dataset_type=celebs_super_resolution \
--exp_dir=./experiment/super_resolution \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1 \
--w_norm_lambda=0.005 \
--resize_factors=1,2,4,8,16,32 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--max_steps=100000 &

nohup python scripts/train.py \
--dataset_type=celebs_super_resolution \
--exp_dir=./experiment/_selftrain_super_resolution \
--workers=8 \
--batch_size=4 \
--test_batch_size=4 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1 \
--w_norm_lambda=0.005 \
--resize_factors=1,2,4,8,16,32 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--max_steps=100000 &
```

RuntimeError: CUDA out of memory.


# 第1步：提取所有人脸特征并存储

用已有模型提取数据集（experiment/generated_yellow-stylegan2，已经下载好，是从http://www.seeprettyface.com/mydataset_page2.html#yellow下载的）中所有人脸特征（10000张），存储在experiment/existing_faces.pkl中，作为人脸特征密钥池

```
nohup python scripts/store_latent_codes.py \
--exp_dir=./experiment/super_resolution_4 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--data_path=./experiment/generated_yellow-stylegan2 \
--test_batch_size=4 \
--test_workers=4 >/dev/null &
```



# 第2步：新人脸匹配的准确率计算及人脸还原

在experiment/Yellow_face/test_faces中有49张经过人脸老化处理的照片，用同样的模型提取特征后，取某几维跟人脸特征密钥池中的特征去匹配，计算匹配的准确度。

可以添加组合方式的有维度选取和匹配标准。

现在的维度选取组合有1维，1-2维，1-3维，...，1-18维，18维，17-18维，16-18维，...，2-18维（当然组合的方式还有很多种，都可以尝试；1维是最高维，18维是最低维，理论上来说，选取一些高维作为匹配依据效果比较好）

现在的匹配标准有余弦相似度(取相似度最高的)和欧拉距离(取距离最小的)，也可以尝试一些其他的匹配标准。

用余弦相似度匹配的是scripts/inference_from_latent.py

用欧拉距离匹配的是scripts/inference_from_latent_distance.py

目前准确率如下所示：（不完全准确，详见“还需要做的事情”部分）

| latent code layers | acc（用余弦相似度） | acc（用欧氏距离） |      |
| :----------------: | :-----------------: | :---------------: | ---- |
|     1(最高维)      |       0.2449        |      0.2449       |      |
|        1-2         |       0.4286        |      0.4082       |      |
|        1-3         |       0.7755        |      0.7551       |      |
|        1-4         |       0.8163        |      0.7959       |      |
|        1-5         |       0.8367        |      0.8367       |      |
|        1-6         |       0.8163        |      0.8163       |      |
|        1-7         |       0.7959        |      0.7959       |      |
|        1-8         |       0.7755        |      0.7551       |      |
|        1-9         |       0.7959        |      0.8163       |      |
|        1-10        |       0.7959        |      0.8163       |      |
|        1-11        |       0.7959        |      0.7959       |      |
|        1-12        |       0.7959        |      0.8163       |      |
|        1-13        |       0.7959        |      0.8163       |      |
|        1-14        |       0.7959        |      0.8163       |      |
|        1-15        |       0.8163        |      0.8163       |      |
|        1-16        |       0.8163        |      0.8163       |      |
|        1-17        |       0.8163        |      0.8163       |      |
|        1-18        |       0.8367        |      0.8367       |      |
|     18(最低维)     |       0.1224        |      0.1224       |      |
|       17-18        |       0.2041        |      0.1837       |      |
|       16-18        |       0.2449        |      0.2449       |      |
|       15-18        |       0.4286        |      0.4286       |      |
|       14-18        |       0.6122        |      0.6122       |      |
|       13-18        |       0.7143        |      0.7143       |      |
|       12-18        |       0.7347        |      0.7347       |      |
|       11-18        |       0.7959        |      0.7959       |      |
|       10-18        |       0.8163        |      0.8163       |      |
|        9-18        |       0.8163        |      0.8163       |      |
|        8-18        |       0.7755        |      0.7755       |      |
|        7-18        |       0.7551        |      0.7347       |      |
|        6-18        |       0.7347        |      0.7347       |      |
|        5-18        |       0.7551        |      0.7347       |      |
|        4-18        |       0.8163        |      0.7755       |      |
|        3-18        |       0.8367        |      0.8367       |      |
|        2-18        |       0.8367        |      0.8367       |      |
|        1-18        |       0.8367        |      0.8367       |      |
|                    |                     |                   |      |
|                    |                     |                   |      |
|                    |                     |                   |      |



杀任务：

```
ps -ef | grep python
kill -9 <id>
```



已经跑过的结果：（每个文件夹中的stats.txt是跑出来的结果）

```
experiment/super_resolution_4 高维+余弦相似
experiment/super_resolution_5 低维+余弦相似
experiment/super_resolution_6 高维+欧氏距离
experiment/super_resolution_7 低维+欧氏距离

```



高维+余弦相似

run_on_batch中有similarity的计算，放开第一个注释就是从高维开始计算

```
nohup python scripts/inference_from_latent.py \
--exp_dir=./experiment/super_resolution_4 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--data_path=./experiment/Yellow_face/test_faces \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs >./record_high_dim.out &

nohup python ./scripts/inference_from_latent.py \
--exp_dir=/root/autodl-tmp/experiment/_super_resolution_1 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--data_path=/root/autodl-tmp/Yellow_face/test_faces \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs >./record_high_dim.out &

nohup python scripts/inference_from_latent.py \
--exp_dir=./experiment/selftrain_super_resolution_1 \
--checkpoint_path=./experiment/selftrain_super_resolution/checkpoints/best_model.pt \
--data_path=./experiment/Yellow_face/test_faces \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs >./record_high_dim.out &

//使用2万步的模型
nohup python scripts/inference_from_latent.py \
--exp_dir=./experiment/selftrain_super_resolution_2 \
--checkpoint_path=./experiment/selftrain_super_resolution/checkpoints/iteration_20000.pt \
--data_path=./experiment/Yellow_face/test_faces \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs >./record_high_dim.out &
```

```
正确的指令
nohup python ./scripts/inference_from_latent.py \
--exp_dir=/root/autodl-tmp/experiment/_super_resolution_1 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--data_path=/root/autodl-tmp/Yellow_face/test_faces \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs >./record_high_dim.out &
```

低维+余弦相似

run_on_batch中有similarity的计算，放开第二个注释就是从低维开始计算

```
nohup python scripts/inference_from_latent.py \
--exp_dir=./experiment/super_resolution_5 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--data_path=./experiment/Yellow_face/test_faces \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs >./record_low_dim.out &
```

```
正确的指令
nohup python ./scripts/inference_from_latent.py \
--exp_dir=/root/autodl-tmp/experiment/_super_resolution_2 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--data_path=/root/autodl-tmp/Yellow_face/test_faces \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs >./record_low_dim.out &
```

高维+欧氏距离

run_on_batch中有distance的计算，放开第一个注释就是从高维开始计算

```
nohup python scripts/inference_from_latent_distance.py \
--exp_dir=./experiment/super_resolution_6 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--data_path=./experiment/Yellow_face/test_faces \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs >./record_high_dim_dis_2.out &
```

```
正确的指令
nohup python ./scripts/inference_from_latent_distance.py \
--exp_dir=/root/autodl-tmp/experiment/_super_resolution_3 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--data_path=/root/autodl-tmp/Yellow_face/test_faces \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs >./record_high_dim_dis_2.out &
```

低维+欧氏距离

run_on_batch中有distance的计算，放开第二个注释就是从低维开始计算

```
nohup python scripts/inference_from_latent_distance.py \
--exp_dir=./experiment/super_resolution_7 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--data_path=./experiment/Yellow_face/test_faces \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs >./record_low_dim_dis_2.out &
```

```
正确的指令
nohup python ./scripts/inference_from_latent_distance.py \
--exp_dir=/root/autodl-tmp/experiment/_super_resolution_4 \
--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
--data_path=/root/autodl-tmp/Yellow_face/test_faces \
--test_batch_size=1 \
--test_workers=1 \
--couple_outputs >./record_low_dim_dis_2.out &
```

想看到还原的效果，可以把图片或者文件夹搬到共享网盘的路径下，然后在平台的“网盘”上查看（网盘的绝对路径是/remote-home/19310044），比如：

```
cp -r ./experiment/super_resolution_4/inference_results/feature_dim_7/inference_coupled/ /remote-home/19310044
```



关于代码细节，可以仔细看一下`scripts/inference.py`（原论文的实现）和`scripts/inference_from_latent.py`（经过修改的实现）的区别，弄清楚Latent code的格式。更多细节可以看看原项目github里的说明。



# 第3步：人脸还原的相似度/Loss计算

人脸相似度得分（identity loss）

```
python ./scripts/calc_id_loss_parallel.py \
--data_path=/root/autodl-tmp/experiment/_super_resolution_1/inference_results/feature_dim_1 \
--gt_path=/root/autodl-tmp/Yellow_face/test_faces_original \
--num_threads=1

python ./scripts/calc_id_loss_parallel.py \
--data_path=/root/autodl-tmp/experiment/_super_resolution_1/inference_results/feature_dim_11 \
--gt_path=/root/autodl-tmp/Yellow_face/test_faces_original \
--num_threads=1
python ./scripts/calc_id_loss_parallel.py \
--data_path=/root/autodl-tmp/experiment/_super_resolution_1/inference_results/feature_dim_12 \
--gt_path=/root/autodl-tmp/Yellow_face/test_faces_original \
--num_threads=1
python ./scripts/calc_id_loss_parallel.py \
--data_path=/root/autodl-tmp/experiment/_super_resolution_1/inference_results/feature_dim_13 \
--gt_path=/root/autodl-tmp/Yellow_face/test_faces_original \
--num_threads=1
python ./scripts/calc_id_loss_parallel.py \
--data_path=/root/autodl-tmp/experiment/_super_resolution_1/inference_results/feature_dim_14 \
--gt_path=/root/autodl-tmp/Yellow_face/test_faces_original \
--num_threads=1
python ./scripts/calc_id_loss_parallel.py \
--data_path=/root/autodl-tmp/experiment/_super_resolution_1/inference_results/feature_dim_15 \
--gt_path=/root/autodl-tmp/Yellow_face/test_faces_original \
--num_threads=1
python ./scripts/calc_id_loss_parallel.py \
--data_path=/root/autodl-tmp/experiment/_super_resolution_1/inference_results/feature_dim_16 \
--gt_path=/root/autodl-tmp/Yellow_face/test_faces_original \
--num_threads=1
python ./scripts/calc_id_loss_parallel.py \
--data_path=/root/autodl-tmp/experiment/_super_resolution_1/inference_results/feature_dim_17 \
--gt_path=/root/autodl-tmp/Yellow_face/test_faces_original \
--num_threads=1
python ./scripts/calc_id_loss_parallel.py \
--data_path=/root/autodl-tmp/experiment/_super_resolution_1/inference_results/feature_dim_18 \
--gt_path=/root/autodl-tmp/Yellow_face/test_faces_original \
--num_threads=1


```



LPIPS loss

```
python scripts/calc_losses_on_images.py \
--mode lpips \
--data_path=/root/autodl-tmp/experiment/_super_resolution_1/inference_results/feature_dim_1 \
--gt_path=/root/autodl-tmp/Yellow_face/test_faces_original \
--workers=1 \
--batch_size=1

python ./scripts/calc_losses_on_images.py \
--mode lpips \
--data_path=./experiment/_super_resolution_1/inference_results/feature_dim_5 \
--gt_path=./experiment/Yellow_face/test_faces_original \
--workers=1 \
--batch_size=1
```



L2 loss

```
python ./scripts/calc_losses_on_images.py \
--mode l2 \
--data_path=/root/autodl-tmp/experiment/_super_resolution_1/inference_results/feature_dim_1 \
--gt_path=/root/autodl-tmp/Yellow_face/test_faces_original \
--batch_size=1 \
--workers=1

```





# 还需要做的事情

## 1 更正测试集

服务器里experiment/Yellow_face/test_faces里大部分图片的标号是正确的，可能有某几(2-3?)张是错误的（比如1176.png）。

更正测试集的流程如下：（手动）

对本地文件夹Yellow_face/test_faces中的每一张已老化图片，在Yellow_face/test_faces_old中找到相同的图片的名称，再到Yellow_face/some_faces中找到该名称对应的图片，作为`图片1`；根据Yellow_face/test_faces中的这张已老化图片的编号，去原数据集generated_yellow-stylegan2找到对应编号的未老化图片，记为`图片2`（可以直接在网盘界面看，觉得加载太慢的话也可以下载下来）。

看`图片1`和`图片2`是否一致，

如果一致，只需要Yellow_face/some_faces中的图片名称改为`对应的编号.png`；

如果不一致，标记一下，暂时不动。

如果排查完之后发现有错的不超过5张，直接在Yellow_face/test_faces和Yellow_face/some_faces中删掉即可（大概率是）；如果有错的太多就再说，可以添加一些数据（老化用的网站是：https://ailab.wondershare.com/tools/aging-filter.html，大部分照片设置的是老化到50岁）。

更正完毕后，把本地的Yellow_face/test_faces和Yellow_face/some_faces上传到192.168.10.6的GPU 1号实例中，用Yellow_face/test_faces覆盖掉原来的experiment/Yellow_face/test_faces，在同目录下上传Yellow_face/some_faces，并改名为test_faces_original。



## 2 使用改好的测试集，重新跑第2步和第3步

之前的数据是基于有一些小错误的测试集跑出来的，不准确，需要重新跑一下



## 3 用我们自己找的的数据集对模型进行fine-tuning，再进行测试

之前的所有测试全部用的是别人训练好的预训练模型，效果不一定最好；把预训练模型在experiment/generated_yellow-stylegan2数据集上微调一下，训练轮数可以多一点（100000steps），训完之后，重复上面的步骤，试试用哪个模型跑出来效果最好(2w轮的/5w轮的/10w轮的)。



## 4 论文写作















