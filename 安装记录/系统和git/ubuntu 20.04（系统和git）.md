***一.  git for ubuntu***

1. **更新软件包列表**：首先，确保你的系统软件包列表是最新的。
    
    ```bash
    sudo apt update
    ```
    
2. **安装Git**：使用以下命令来安装Git。
    ```bash
    sudo apt install git
    ```

3. **验证安装**：安装完成后，你可以验证Git是否安装成功，并查看其版本。
    
    ```bash
    git --version
    ```
    
    如果安装成功，你应该会看到类似于 `git version 2.x.x` 的输出。
    
4. **配置Git**：在你开始使用Git之前，建议配置你的用户名和电子邮件，这对于提交记录是很重要的。
    ```bash
    git config --global user.name "Your Name"
    git config --global user.email "your.email@example.com"
    ```

5. **（可选）配置其他选项**：你还可以配置其他选项，比如设置默认的文本编辑器。
    ```bash
    git config --global core.editor nano  # 例如设置nano为默认编辑器
    ```

6. **检查配置**：你可以查看所有Git的配置。
    ```bash
    git config --list
    ```

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



***二. cuda for ubuntu***



### 1. **更新系统**

在开始之前，确保你的系统是最新的。
```bash
sudo apt update
sudo apt upgrade
```

### 2. **安装必要的软件包**

CUDA安装过程中需要一些依赖包，确保这些包已经安装。
```bash
sudo apt install build-essential dkms
```

### 3. **添加NVIDIA包存储库**

首先，添加NVIDIA的包存储库，以便可以获取最新的CUDA工具包。

```bash
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 7FA2AF80
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
```

### 4. **安装NVIDIA驱动程序**

安装适用于你的GPU的NVIDIA驱动程序。

```bash
sudo apt update
sudo apt install nvidia-driver-<version>
```

请替换`<version>`为你要安装的驱动版本，例如`nvidia-driver-470`。你可以通过以下命令查看可用的版本：
```bash
apt search nvidia-driver
```

### 5. **重启系统**

安装完驱动程序后，重启系统以应用更改。
```bash
sudo reboot
```

### 6. **下载并安装CUDA工具包**

前往[NVIDIA的CUDA下载页面](https://developer.nvidia.com/cuda-downloads)下载适用于Ubuntu 20.04的CUDA工具包。

选择你需要的CUDA版本，下载`.deb`安装包。例如，如果你选择的是`CUDA 11.8`，你可以下载类似`cuda-repo-ubuntu2004-11-8-local_11.8.0-1_amd64.deb`的文件。

然后安装下载的包：
```bash
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-1_amd64.deb
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 7FA2AF80
sudo apt update
sudo apt install cuda
```

### 7. **设置环境变量**

为了让系统能够找到CUDA工具，设置环境变量。你可以在`~/.bashrc`中添加如下行：

```bash
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

然后，使这些更改生效：

```bash
source ~/.bashrc
```

### 8. **验证CUDA安装**

重启后，检查CUDA是否正确安装。你可以运行以下命令来查看CUDA版本：

```bash
nvcc --version
```

### 9. **（可选）安装CUDA示例**

你可以安装CUDA的示例代码来测试CUDA是否正确工作：

```bash
cuda-install-samples-11.8.sh ~
cd ~/NVIDIA_CUDA-11.8_Samples
make
```

完成后，你可以运行一些示例程序来测试CUDA安装。

### 10. **安装cuDNN（可选）**

如果你需要cuDNN支持，可以从[NVIDIA cuDNN页面](https://developer.nvidia.com/cudnn)下载并安装。

下载cuDNN后，解压并将文件复制到CUDA目录中：

```bash
tar -xzvf cudnn-<version>-linux-x64-v<version>.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 官方地址
https://docs.nvidia.com/cuda/

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

***三. cuddn for ubuntu 20.04***



### 1. **下载cuDNN**

1. 访问[NVIDIA cuDNN下载页面](https://developer.nvidia.com/cudnn)。
2. 注册并登录到NVIDIA开发者网站。
3. 选择适用于你的CUDA版本的cuDNN版本。例如，如果你使用的是CUDA 11.8，选择相应的cuDNN版本。
4. 下载对应的Linux版本的`tar`文件，例如`cudnn-11.8-linux-x64-v8.9.1.34.tgz`。

### 2. **解压cuDNN文件**

下载完成后，使用以下命令解压cuDNN文件。假设文件名为`cudnn-11.8-linux-x64-v8.9.1.34.tgz`，你可以使用以下命令：

```bash
tar -xzvf cudnn-11.8-linux-x64-v8.9.1.34.tgz
```

这会创建一个包含`cuda`目录的解压缩目录。

### 3. **将cuDNN文件复制到CUDA目录**

将解压后的cuDNN文件复制到CUDA安装目录中。假设你的CUDA安装在`/usr/local/cuda`目录下，可以使用以下命令：

```bash
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
```

### 4. **设置正确的权限**

确保cuDNN库文件具有适当的权限，以便可以被系统访问：

```bash
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 5. **验证cuDNN安装**

你可以通过编写和运行一个简单的测试程序来验证cuDNN是否正确安装。以下是一个基本的测试步骤：

1. **创建一个测试程序**：

   创建一个文件 `test_cudnn.cpp`，内容如下：

   ```cpp
   #include <cudnn.h>
   #include <iostream>

   int main() {
       cudnnHandle_t handle;
       cudnnCreate(&handle);
       std::cout << "cuDNN version: " << CUDNN_VERSION << std::endl;
       cudnnDestroy(handle);
       return 0;
   }
   ```

2. **编译和运行测试程序**：

   ```bash
   g++ test_cudnn.cpp -o test_cudnn -lcudnn
   ./test_cudnn
   ```

   如果cuDNN正确安装，你应该能看到类似`cuDNN version: 8900`（版本号取决于你安装的cuDNN版本）的输出。

### 官网
 https://docs.nvidia.com/deeplearning/cudnn/