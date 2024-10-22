#vex-ai-demo和tju-vex库合并

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729614574974.png)

包括好一起的库

	#include "ai_functions.h"

原来的库没有自动包含，需要在main.cpp里面加入一段

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729614650486.png)

	ai::jetson  jetson_comms;
	static AI_RECORD local_map;

