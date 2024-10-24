#格斗机器人ai、程序

##程序

程序的控制编程主要由vex v5官方提供的vscode编译扩展实现，然后通过cpp集成编程。

项目集成如下：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729708175776.png)


###底盘控制

底盘使用了pid的闭环控制原理控制车型走直线，PID（比例-积分-微分）控制是一种常见的控制算法，它广泛应用于工业过程控制、机器人控制、自动驾驶等领域。PID控制的基本原理是根据被控对象的当前状态与设定值之间的差异（即误差）来调整输出信号，使得差异趋近于零，实现稳定精准的自动控制。

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729708668937.png)

PID控制器的工作原理包括传感器检测、PID计算、输出调节和反馈控制四个步骤：

传感器检测：系统首先通过传感器实时检测被控对象的状态，并将其与设定值进行比较，得到误差。

PID计算：PID控制器根据比例、积分和微分三个部分的权重，将误差转换为输出信号。具体计算方式为：输出信号=Kpe(t)+Ki∫e(t)dt+Kd*de(t)/dt，其中Kp、Ki和Kd分别为比例、积分和微分系数，e(t)为误差，∫e(t)dt为误差的积分，de(t)/dt为误差的微分。

输出调节：PID控制器将计算得到的输出信号传递给执行器，执行器根据该信号调节被控对象的状态。这可以是通过改变电流、压力、速度等方式来实现。

反馈控制：系统不断重复上述步骤，通过不断调整输出信号，使得被控对象的状态逐渐接近设定值。同时，通过传感器不断监测被控对象的实际状态，实现反馈控制，进一步修正输出信号，以实现更加精确的控制。

在本机器人的地盘移动函数中也同样使用了pid函数

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729708626171.png)

###功能函数

在机器人的特殊功能中，特定的实现即可。

##ai

###基本detection

在基本的objection detection部分，我们使用了jetson nano b01 基础版的计算版，搭配D435，基于双目的深度相机，配合上yolo系列训练好的objection detection的pt模型转化为onnx推理模型部署到jetson nano中调用，ONNX是一种开放的神经网络交换格式，它定义了一组与环境、平台均无关的标准格式，用于存储训练好的模型，包括模型的权重、结构信息以及每一层的输入输出等，旨在解决不同框架和硬件平台之间的互操作性问题，使得模型可以在不同的深度学习框架和硬件平台之间自由迁移和部署。

###yolo

我们自己训练了2000+的训练集和300+的验证集，来训练yolo模型

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729706446410.png)

在100为一个单位的batch训练了300个epoch后，损失函数和map趋近收敛，平均精度达到83以上，detection的可视化如下

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729706554790.png)

仍然存在的detection问题就是置信度太高，导致模型处理的result返还速度下降

###detection的python部署



在借助yolo集成好的库中电泳preprocessyolo和postprocessyolo模型实现数据的接受传递预处理

然后从onnx中调用get_engine函数讲推理模型onnx中构造好的参数实现model构造，然后把构造的model交付给dataprocess.py进行进一步使用

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729707622905.png)

Model类

get_engine方法：这是一个静态方法，用于尝试加载一个已经存在的TensorRT引擎文件。如果文件不存在，它会从ONNX文件构建一个新的引擎，并将其保存到指定的路径。这个方法首先检查引擎文件是否存在，如果存在则加载它；如果不存在，则通过以下步骤构建引擎：
创建一个TensorRT构建器、网络、配置和ONNX解析器。
设置最大工作空间大小和批量大小。
检查ONNX文件是否存在，然后解析它。
设置网络的输入形状。
构建网络并序列化，然后创建并返回引擎。

__init__方法：初始化模型实例，加载TensorRT引擎，创建执行上下文，并分配输入和输出的缓冲区。

inference方法：对给定的图像执行推理，返回检测到的对象的边界框、分数和类别。这个方法首先使用PreprocessYOLO类处理输入图像，然后执行推理，并使用PostprocessYOLO类处理输出，最后绘制边界框并返回处理后的图像和检测结果。

draw_bboxes方法：这是一个静态方法，用于在原始图像上绘制边界框，并返回修改后的图像。这个方法还会将检测到的对象信息存储在一个列表中。

rawDetection类

这个类用于存储检测到的对象的信息，包括x和y坐标、中心点、宽度、高度、概率和类别ID。

###数据处理

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729707501847.png)

在dataprocessing.py中，我们调用了you only look once （YOLO）模型的经典思想，对象框提议，然后实现一个基础的非极大抑制（NMS），NMS是一种后处理技术，用于在目标检测过程中去除多余的边界框，从而提高检测的准确性。在目标检测中，经常会出现多个重叠的候选边界框，它们可能对应于同一个目标。NMS通过保留最佳的边界框并抑制其他重叠框来解决这个问题。考虑到本场景中会出现多个目标叠加在一起的情况，我们把NMS后处理放在首要的位置。

load_label_categories 函数

目的：从指定的文本文件（labels.txt）中加载类别名称。
参数：label_file_path - 类别名称文件的路径。
返回值：包含所有类别名称的列表。

PreprocessYOLO 类

目的：处理输入图像，使其适合YOLOv3模型的输入要求。

方法：
__init__：初始化类，设置YOLOv3的输入分辨率。

process：处理输入图像，包括加载、调整大小和归一化，然后返回原始图像和处理后的图像。

_load_and_resize：加载图像并调整其大小到指定的输入分辨率。

_shuffle_and_normalize：将图像数据归一化到[0, 1]范围，并调整数据布局以符合模型要求。

PostprocessYOLO 类

目的：处理YOLOv3模型的输出，提取检测到的对象的边界框、类别和置信度。
方法：

__init__：初始化类，设置YOLOv3的掩码、锚点、对象阈值、NMS阈值和输入分辨率。

process：处理模型的输出，返回检测到的对象的边界框、类别和置信度。

_reshape_output：将模型的输出重新整形为适合进一步处理的形式。

_process_yolo_output：处理所有输出，包括应用NMS算法以减少重叠的边界框。

_process_feats：计算每个网格单元中检测到的边界框、置信度和类别概率。

_filter_boxes：根据对象阈值过滤边界框。

_nms_boxes：应用非最大抑制（NMS）算法，以减少相邻且重叠的边界框数量。

关键点

YOLOv3输出处理：YOLOv3模型输出三个不同尺度的特征图，每个特征图对应不同大小的对象。这些输出需要进一步处理以提取有用的信息（如边界框、类别和置信度）。

非最大抑制（NMS）：是一种用于减少检测到的边界框数量的技术，特别是当多个边界框重叠时。它保留具有最高置信度的边界框，并删除与之重叠的、置信度较低的边界框。

数据归一化和重新排序：输入图像需要被调整大小、归一化，并重新排序以符合YOLOv3模型的输入要求。同样，模型的输出也需要被重新整形和处理以提取有用的信息


###检测可视化

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729707412703.png)

随后我们考虑方便对机器进行调试，实现了dashboard可视化。

代码定义了一个名为 V5WebData 的类，它是一个用于处理与 WebSocket 客户端通信的服务端应用程序。这个类主要用于在一个指定的端口上启动一个 WebSocket 服务器，从而允许客户端通过 WebSocket 连接接收和发送数据。这个服务器主要用于处理和传输与机器人或类似设备相关的实时数据，如图像、位置、检测结果、统计信息等。下面是代码的主要组成部分及其功能的详细解释：

类 Statistics

功能：用于存储和报告实时统计信息，如帧率（FPS）、推理时间、CPU温度、视频宽度和高度、运行时间以及GPS连接状态。

类 V5WebData

构造函数：

初始化服务器端口、加载GPS和相机的偏移量、颜色校正值，并创建WebSocket服务器实例。

start 方法：启动WebSocket服务器，并初始化用于存储检测数据、图像和统计信息的变量。

__new_client 方法：当有新客户端连接时调用，打印客户端ID。

__client_left 方法：当客户端断开连接时调用，打印断开连接的客户端ID。

__message_received 方法：处理从客户端接收到的消息。根据消息内容，它可能更新GPS或相机偏移、颜色校正值，或者返回请求的数据（如位置、检测、统计、图像等）。

setDetectionData、setColorImage、setDepthImage、setStatistics 方法：这些方法用于更新检测数据、彩色图像、深度图像和统计信息。

setGpsOffset、setCameraOffset、setColorCorrection 方法：这些方法用于更新GPS偏移、相机偏移和颜色校正值，并将更新保存到JSON文件中。

isConnected 方法：检查是否有客户端连接。

stop 方法：优雅地关闭服务器。

__del__ 方法：析构函数，当对象被删除时调用stop方法。

类 Offset、GPSOffset、CameraOffset

功能：这些类用于表示不同类型的偏移量（GPS偏移和相机偏移）。GPSOffset和CameraOffset继承自Offset类，并添加了特定的属性（如CameraOffset中的elevation_offset）。这些类还提供了从JSON文件加载和保存偏移量的方法。

类 ColorCorrection

功能：用于存储和处理颜色校正值（HSV值）。提供了从JSON文件加载和保存颜色校正值的方法。

辅助方法

__getStatsElement、__getPositionElement、__getDetectionElement、__getColorElement、__getDepthElement：这些方法用于获取当前统计信息、位置、检测数据、彩色图像和深度图像的元素，以便通过WebSocket发送给客户端。
convert_numpy_to_list：一个辅助函数，用于递归地将NumPy数组转换为列表，以便可以将数据作为JSON发送。

###主控通信

在jetson nano 主板上处理完检测到的目标消息后，要和模拟环境中规定使用的v5主控实现通。

在硬件连接中，使用有线通信将jetson nano和v5主板连接实现通信，这样通信的效率能够满足detection的处理需要

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241024024043.png)

在v5主板上实现控制，此时必须使用cpp编程，好在有request map函数对上文提到的检测到的数据** rawDetection类** 去保存成为本地的一个local map变量

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729708737328.png)

local_map的变量类型是AI_RECORD 里面包含了检测的物体的类型，位置

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729709015854.png)

###决策

此时基础控制和感知都实现了，那么可以考虑一下在主板上执行决策问题了。

我们模拟实现的是一个基于vex的搬运问题。那么我们考虑这是一个带有回归的基于旅行商问题的算法优化。我们暂定场上需要搬运的物体数量级是1e2级别的，复杂度分析来看时间复杂度就是

O(2^n*n^2)也就是接近每次处理1e7级别的数据，然而jetson nano的算力达到1e10级别，每秒钟能够实现1000次的该运算，因此算力不是问题 

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729709681515.png)

那么在决策的算法可以暂定为TSP问题，接下来我们简单看看初始的TSP问题：假设有一个旅行商人要拜访N个城市，他必须选择所要走的路径，路径的限制是每个城市只能拜访一次，而且最后要回到原来出发的城市。路径的选择目标是要求得的路径路程为所有路径之中的最小值。TSP问题是一个NPC问题。

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241024025729.png)

从自己所在的点开始的最短遍历路径则是1–>4–>3–>2–>1

那么这个问题如何扩展为本项目的决策：可以把点换为我们需要去取到的物品，边权转化为我们两点之间的距离（因为距离的比例永远不变，所以可以直接用长度m为单位作为他们的边权w），然后整个图应该是一个完全图（因为你可以从任意一个点出发去到另外一个点去做走）

就拿这个为例子，我们执行一个动态规划

1、状态压缩

  我们需要表达我们已经走过了哪些点，目前到达了哪里，有什么办法表达出来呢？

  暴力是万能的，我们可以开一个数组dp[i][j]，代表目前到达了i点，dp[i][j]的值代表j点是否已经走过了，但是这样做的话我们状态转移会变得很麻烦，状态压缩就是它的优化

  状态压缩是通过二进制实现的，我们知道int有32位，那么我们可以用第0位代表第0个点的状态，第1位代表第1个点状态…第n位代表第n个点的状态，位的值如果是1的话就代表该点已经走过了，例如17的二进制为0000010001，代表第0个点和第4个点已经走过了

  那么我们可以开一个数组dp[i][j]，代表目前走到了i点，用j代表已经走过了哪些点，例如：
dp[0][17]，17的二进制为0000010001,代表目前在第0个点，已经走过第0个点和第4个点。
dp[4][17]，17的二进制为0000010001,代表目前在第4个点，已经走过第0个点和第4个点。

  我们可以用dp[i][j]的值代表当前这个状态的最小花费，例如dp[0][17]=12，那么就代表到达该状态需要的最小花费是12

2、状态转移
  dp的基本思想就是记录某个状态的最优解，再从目前的状态转移到新的状态，从局部最优解转移到全局最优解

  我们用数组a[i][j]存储图，那么a[i][j]的值就代表从i点到j点的花费

  我们如何求状态dp[0][19]的最优解？
  19的二进制是0000010011，因为18的二进制为0000010010，那么dp[0][19]可以由dp[4][18],dp[1][18]转移过来，最小花费是dp[0][19]=min(dp[4][18]+a[4][0],dp[1][18]+a[1][0])

  即我们要求大的状态，那么就需要先把小状态最优解求出来。反过来我们求出了所有小状态，那么就可以求出大状态的最优解

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241024030116.png)

{1，2，3}代表第1、2、3个点都已经走过了

  可以发现，小状态总是比大状态小的，那么我们可以从0状态枚举到2n-1状态，获取到每个状态的最优解

我们还可以反过来想，从小状态去更新大的状态

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241024030138.png)

落实到具体的代码则是先要实现一个从local_map中获取物体存储在自己的数据结构中的操作

	const int MAX_NUM_RINGS = 15;
	double a[MAX_NUM_RINGS][MAX_NUM_RINGS];
	double dp[MAX_NUM_RINGS][1<<MAX_NUM_RINGS];
	int t;
	//自建图
	struct ring
	{
	    int col;//0--red, 1--blue
	    double x,y;
	};
	std::vector<ring> R;
	void create_map(double diff = 0.05,AI_RECORD local_map)
	{
	    int num = R.size();
	    for(int j=0;j<local_map.detectionCount;j++)
	    {    
	        double nx = local_map.detections[j].mapLocation.x+local_map.pos.x;
	        double ny = local_map.detections[j].mapLocation.y+local_map.pos.y;
	        int col = local_map.detections[j].classID;
	        for(int i=0;i<num;i++)
	        {
	            if(abs(nx-R[i].x)>diff || abs(ny-R[i].y)>diff)
	            {
	                R.push_back({col,nx,ny});
	                break;
	            }
	        }
	    }
	}

其中，比对目前数据结构和local_map中每一个物体位置，差距如果大于diff，那么就判断他为一个新的物体

这个diff的选择是有意义的，如果太小，那么很容易把有误差的检测同一个物体判断为两个不同的物体，如果太大，那么会导致两个邻近的物体被判断为不存在，那么就漏掉了物体。

所以我们对diff执行了一个枚举，跑了几次RUN，然后记录漏下的物体数量，效果如下

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729710824177.png)

因此最后diff选择了0.05

然后初始化所有已经检测到的点

	void init_dp()
	{
	    int n=R.size();
	    int m=n*(n-1)/2;//完全图的边数
	    memset(dp,0,sizeof(dp));
	    memset(a,0,sizeof(a));
	    for(int i=0;i<m;i++)
	    {
	        for(int j=0;j<m;j++)
	        {
	            a[i][j]=sqrt((R[i].x-R[j].x)*(R[i].x-R[j].x)+(R[i].y-R[j].y)*(R[i].y-R[j].y));
	            a[j][i]=a[i][j];
	            //双向边
	        }
	    }
	}

注意是双向边

	void run()
	{
	    //dp核心算法
	    int n=R.size();
	    int m=n*(n-1)/2;//完全图的边数 
		t=(1<<n);
		for(int i=1;i<n;i++){//因为起点初始不能被标记已经走过,所以需要手动初始化起点到达其它点的花费 
			dp[i][1<<i]=a[0][i];
		}
		for(int i=0;i<t;i++){//枚举每一个状态 
			for(int j=0;j<n;j++){//枚举每一个没有走过的点 
				if(((i>>j)&1)==0){
					for(int k=0;k<n;k++){//枚举每一个走过的点 
						if(((i>>k)&1)==1&&dp[j][i^(1<<j)]>dp[k][i]+a[k][j]){//取最优状态 
							dp[j][i^(1<<j)]=dp[k][i]+a[k][j];
						}
					}
				}
			}
		}
	}

最后模拟之前图示的动态规划

	int tt;//记录 
	std::vector<int> path(1,0);//初始化从0点出发 ,存储单条路径 
	std::vector<std::vector<int> > paths;//存储所有的路径 
	void getPath(int p){//递归查找所有路径 
	    int n=R.size();
	    int m=n*(n-1)/2;//完全图的边数 
		if((tt^(1<<p))==0){//如果是最后一个点了就存储改路径 
			paths.push_back(path);
			return; 
		}
		for(int j=1;j<n;j++){
			//回溯算法，一个加法的原则
			//如果点1到达点5的最短距离为100，点1到达点3的最短距离是70
			//而点3和点5之间的距离为30 ，那么点3是点1到5之间的一个中间点
			//即1-->...-->3-->5 
			if(a[j][p]+dp[j][tt^(1<<p)]==dp[p][tt]){
				tt^=(1<<p);
				path.push_back(j);
				getPath(j);
				tt^=(1<<p);
				path.pop_back();
			}
		}
		
	}

然后把路线记录在paths中

注意的是，我们机器人是实时移动更新数据，在while循环中不断获得信息，不断从jetson nano中获得数据，更新自己的数据结构。

	   {
        create_map(0.3,local_map);
        init_dp();
        run();
        tt=0;
        path.clear();
        getPath(0);
        int target_ring=paths[0][paths[0].size()-1];
        double timecost=sqrt(pow(R[target_ring].x-local_map.pos.x,2)+pow(R[target_ring].y-local_map.pos.y,2));
        ODrive.moveToTarget({R[target_ring].x,R[target_ring].y,0});
        R[target_ring].x=1e5+5;
        R[target_ring].y=1e5+5;
        // 等待到达目标点
        this_thread::sleep_for(timecost*1000);
      }  

注意每次决策执行

	ODrive.moveToTarget({R[target_ring].x,R[target_ring].y,0});

之后，要把取到的点直接“抛出地图”，设置成1e5+5；

###执行

###应用