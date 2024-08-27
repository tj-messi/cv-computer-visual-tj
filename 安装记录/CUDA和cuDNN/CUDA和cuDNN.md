#CUDA 和 cuDNN 安装

---
##CUDA安装

安装11.3.0版本的CUDA

打开https://developer.nvidia.com/cuda-toolkit-archive

再进行wget

wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run

chmod 775 ./cuda_11.3.0_465.19.01_linux.run

sudo ./cuda_11.3.0_465.19.01_linux.run

输入accept  回车
	
把Driver的X去了（空格键回车）
	
install 回车

vim ~/.bashrc

export CUDA=/usr/local/cuda
	
export LD_LIBRARY_PATH=$CUDA/lib64:$LD_LIBRARY_PATH
	
export PATH=$CUDA/bin:$PATH

nvcc -V 检查cuda的版本

---
##cuDNN安装

https://developer.nvidia.com/cudnn-downloads
	
下载8.X版本
	
https://developer.nvidia.com/rdp/cudnn-archive
	
8.9.5   11.X
	
	 
tar -xvf cudnn-linux-x86_64-8.9.5.30_cuda11-archive.tar.xz 
	 
cd  cudnn-linux-x86_64-8.9.5.30_cuda11-archive/
	 
chmod -R 775 ./*
	 
	 
sudo mv ./include/* /usr/local/cuda/include/
	 
sudo mv ./lib/* /usr/local/cuda/lib64/
	 
	
--检查cudnn版本 
	
cat /usr/local/cuda/include/cudnn_version.h
	
检查
	
define CUDNN_MAJOR 8
define CUDNN_MINOR 9
define CUDNN_PATCHLEVEL 5

define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
