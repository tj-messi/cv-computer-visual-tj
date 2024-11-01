#pragma once
// 通用pid控制�?
#include "pidControl.hpp"
// 基�?�函数
#include "Math-Functions.h"
// 基�?�底盘类：底盘运动控制算法
#include "tjulib-chassis/basechassis.hpp"
#include "tjulib-chassis/ordinary-chassis/ordi-chassis.hpp"
#include "tjulib-chassis/oct-chassis/oct-chassis.hpp"
// 上层结构：上层机械结构件控制
#include "tjulib-uppercontrol/ConveyRing.hpp"
#include "tjulib-uppercontrol/LiftArm.hpp"
// 远程调试类：使用手柄及蓝牙连接主机进行本地调�?
#include "RemoteDebugSerial.hpp"
// 定位策略
#include "tjulib-position/PositionStrategy.hpp"
// 规划
#include "tjulib-motionplanner/tjulib-actionplanner/purepursuit.hpp"
#include "tjulib-motionplanner/tjulib-pathplanner/rrt.hpp"
//HS地图
#include "tjulib-map/HighStakeMap.hpp"
// vex ai
#include "vex-ai/ai_robot_link.h"

