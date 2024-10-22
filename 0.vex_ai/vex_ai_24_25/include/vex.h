#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "v5.h"
#include "v5_vcs.h"

#include "robot-config.h"
#include "vex-ai/ai_jetson.h"
#include "vex-ai/ai_robot_link.h"

#include <vector>

extern ai::jetson      jetson_comms;
extern ai::robot_link  link;
extern int dashboardTask( void );

#define waitUntil(condition) \
  do                         \
  {                          \
    wait(5, msec);           \
  } while (!(condition))

#define repeat(iterations) \
  for (int iterator = 0; iterator < iterations; iterator++)