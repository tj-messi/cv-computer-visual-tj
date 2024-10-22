#vex-ai-demo和tju-vex库合并

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729614574974.png)

包括好一起的库

	#include "ai_functions.h"

原来的库没有自动包含，需要在main.cpp里面加入一段

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729614650486.png)

	ai::jetson  jetson_comms;
	static AI_RECORD local_map;

命名一个jetson的实例，命名一个AI_RECORD的实例

然后ai-demo的运行逻辑是，第一次进入Isolation的线程之后，第二次进入Interaction线程，代码如下

	void auto_Isolation(void) { 
	  
	  //test code
	  while(1){
	    //if(local_map.detectionCount>0)
	    {
	      Arm.spin(fwd);
	    }
	  }
	
	}
	
	void auto_Interaction(void) {
	
	  if(local_map.detectionCount>0)
	  {
	      Arm.spin(fwd);
	  }
	  // Add functions for interaction phase
	}
	bool firstAutoFlag = true;
	
	void autonomousMain(void) {
	
	  if(firstAutoFlag)
	    auto_Isolation();
	  else 
	    auto_Interaction();
	
	  firstAutoFlag = false;
	}

注意competition实例的命令

	// A global instance of competition
	competition Competition;

！不能使用原来的automatic的competition类
