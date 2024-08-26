#固态硬盘安装和linux双系统

---
##固态硬盘安装
使用天选四安装固态硬盘的时候就注意好断电处理即可

安装好固态硬盘之后要win+x键的磁盘管理中能看到一个950多gb未初始化的磁盘

---
##linux系统安装
参考文档：https://blog.csdn.net/weixin_44415639/article/details/131934907

---
###在SSD上完全安装linux
先下载ubuntu的20.04镜像，通过烧录到u盘中制作启动盘

再格式化硬盘的分区

通过磁盘管理检查自己盘的类型：属性-卷-GPT分区形式

记得每次都要关闭VMD模式，从而进行自己选择盘来让ubuntu系统检测到新安装的1TB固态硬盘

---
###ubuntu系统的初始化
在BIOS系统中选择了u盘启动的之后，就可以进入u盘装载的启动盘了。随后选择something else从而进行自行分配盘。

---
###ubuntu系统的手动分盘
1.先分配一个大概32000mb的盘，use as：swap area

2.之后把整个盘都分配给ext4 journaling file system

使用mount point ：/

