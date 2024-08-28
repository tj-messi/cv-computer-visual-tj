#conda安装

---
##conda的安装步骤
在https：//mirrors.tuna.tsinghua.edu.cn/anaconda/archive/镜像源中下载合适版本的conda。

这里选择2022.05-linux-x86-64.sh

cd downloads之后mv 2022.05-linux-x86-64.sh的包 /home/vex/

cd ..

pwd /home/vex

./

sudo chmod 777 包

mv 2022.05-linux-x86-64.sh的包 /home/vex/^C

./2022.05-linux-x86-64.sh的包

这样就在主目录创造了一个anaconda3的文件夹里面开始安装conda

---
##拉取torch
搜索pytorch官网https://pytorch.org

输入conda create -n py36 python=3.6.8

然后conda activate py36

然后conda install pytorch==1.11.0 torchvision==0.12.0 tochaudio==0.11.0 cudatoolkit==11.3 -c pytorch

新开一个控制台 

检查一下pip --version

然后pip install opencv-python collection opencv-python

最后安装对应版本的conda

参考那个1分钟的视频

