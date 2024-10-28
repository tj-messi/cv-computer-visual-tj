#include "vex.h"

using namespace vex;

/*************************************

        physical configurations

*************************************/
const double car_width = 11.15;                   // 轮距
const double r_wheel = 4 / 2;                     // 车轮半径
const double gear_ratio = 0.667;                  // 底盘电机-轮的齿轮传动比（加速配比就大于1，减速配比就小于1）
const double r_motor = r_wheel * gear_ratio ;     // 电机转角-电机转周的换算比
const double cell  = 24;                          // 一个地垫长度(inches)
const double hOffset  = -5;                       // 里程计偏置（inches）----从旋转中心向里程计轮延伸方向作垂线
const double vOffset  = 5;                        // 里程计偏置（inches）----从旋转中心向里程计轮延伸方向作垂线
const double r_wheel_encoder = 2.75 / 2;          // 编码轮周长
double gps_offset_x = 0;                    // GPS的x轴方向偏置 
double gps_offset_y = 6.7;                  // GPS的y轴方向偏置 
const double encoder_rotate_degree = 45;          // 编码轮旋转角度
double camera_offset_x = 0;                    // 摄像头的x轴方向偏置 
double camera_offset_y = 6.7;                  // 摄像头的y轴方向偏置 
/*************************************

            state flags

*************************************/
// gps坐标(带旋转中心偏置修正)
double gps_x = 0;
double gps_y = 0;
double gps_heading = 0;

double gps_x_small = 0;
double gps_y_small = 0;

bool manual = false;             // 是否是手动
bool reinforce_stop = false;    // 是否强制终止吸环
bool ring_convey_spin = false;   // 是否开始进行吸环
bool photo_flag = false;         // 是否需要进行拍照 
bool ring_convey_stuck = false;  // 是否卡环
int ring_color = 0;              // 对获取的环的颜色进行检查，0是没有环，1是要保留的环，2是要丢弃的环
bool half_ring_get = false;     // 半吸环

/*************************************

            VEX devices

*************************************/
// A global instance of brain used for printing to the V5 Brain screen
brain  Brain;
// 底盘电机 - 四角底盘
motor L1 = motor(PORT5, ratio6_1, false);
motor L2 = motor(PORT5, ratio6_1, true);
motor L3 = motor(PORT5, ratio6_1, false);
motor L4 = motor(PORT5, ratio6_1, true);
motor R1 = motor(PORT5, ratio6_1, true);
motor R2 = motor(PORT5, ratio6_1, false);
motor R3 = motor(PORT5, ratio6_1, true);
motor R4 = motor(PORT5, ratio6_1, false); 
std::vector<vex::motor*> _leftMotors = {&L1, &L2, &L3,  &L4};
std::vector<vex::motor*> _rightMotors = {&R1, &R2, &R3, &R4};

// 底盘电机 - 八角底盘
motor lf1 = motor(PORT1, ratio18_1, false);
motor lf2 = motor(PORT2, ratio18_1, true);
motor lb1 = motor(PORT3, ratio18_1, false);
motor lb2 = motor(PORT4, ratio18_1, true);
motor rf1 = motor(PORT10, ratio18_1, false);
motor rf2 = motor(PORT9, ratio18_1, true);
motor rb1 = motor(PORT11, ratio18_1, false);
motor rb2 = motor(PORT12, ratio18_1, true);

std::vector<vex::motor*> _lfMotors = {&lf1, &lf2};
std::vector<vex::motor*> _lbMotors = {&lb1, &lb2};
std::vector<vex::motor*> _rfMotors = {&rf1, &rf2};
std::vector<vex::motor*> _rbMotors = {&rb1, &rb2};

// 抬升臂
motor lift_armMotorA = motor(PORT8, ratio36_1, true);
motor lift_armMotorB = motor(PORT15, ratio36_1, false);
motor_group lift_arm = motor_group(lift_armMotorA, lift_armMotorB);

// 传送带
motor convey_beltMotorA = motor(PORT18, ratio36_1, true);
motor convey_beltMotorB = motor(PORT14, ratio36_1, false);
motor_group convey_belt = motor_group(convey_beltMotorA, convey_beltMotorB);

// 吸球
motor rollerMotorA = motor(PORT6, ratio18_1, true);
motor rollerMotorB = motor(PORT5, ratio18_1, false);
motor_group roller_group = motor_group(rollerMotorA, rollerMotorB);
// 遥控器
controller Controller1 = controller(primary);
// 通信天线
vex::message_link AllianceLink(PORT20, "tju1", linkType::worker);
// 里程计
encoder encoderHorizonal = encoder(Brain.ThreeWirePort.A);
encoder encoderVertical = encoder(Brain.ThreeWirePort.G);
// 导入杆
motor side_bar = motor(PORT5, ratio18_1, false);
// imu惯性传感器
inertial imu = inertial(PORT17);  // 第二个参数要写right

// 气动件
pwm_out gas_push = pwm_out(Brain.ThreeWirePort.D);
pwm_out gas_lift = pwm_out(Brain.ThreeWirePort.E);
pwm_out gas_hold = pwm_out(Brain.ThreeWirePort.F);

// 距离传感器
distance DistanceSensor = distance(PORT13);
// gps
gps GPS_ = gps(PORT16, 0, 0, inches, 0);

// vision signature
vision::signature Red1 = vision::signature(1, 9051, 11375, 10213, -1977, -833, -1405, 1.6711680, 0);
vision::signature Red2 = vision::signature(1,  5461, 8761, 7111, -1457, -167, -812, 0.8144962, 0);
vision::signature Red3 = vision::signature(1,  8191, 9637, 8914, -1831, -735, -1283, 0.6828844, 0);

vision::signature Blue1 = vision::signature(2, -5461, -3919,  -4690, 7721, 11045, 9383, 1.123142, 0);
vision::signature Blue2 = vision::signature(2, -5461, -3919,  -4690, 7721, 11045, 9383, 1.123142, 0);
vision::signature Blue3 = vision::signature(2,  -4177, -3545, -3861, 6099, 7047, 6573, 1.2, 0);

std::vector<vision::signature*>Red = {&Red1, &Red2, &Red3};
std::vector<vision::signature*>Blue = {&Blue1, &Blue2, &Blue3};

vision::signature Stake_Red = vision::signature(1,  6889, 10071, 8480, -1219, -343, -781, 0.7551798, 0);
vision::signature Stake_Blue = vision::signature(2,  -5461, -3919,  -4690, 7721, 11045, 9383, 1.123142, 0);
vision::signature Stake_Yellow = vision::signature(3,   -3313, -1611, -2462, -6767, -5311, -6039, 0.9483626, 0);
// vision
vision Vision = vision(PORT19, 50, Red1, Blue2);
vision Vision_front = vision(PORT5, 50, Stake_Red, Stake_Blue, Stake_Yellow);


// VEXcode generated functions
// define variable for remote controller enable/disable
bool RemoteControlCodeEnabled = true;

// manager and worker robots
// Comment out the following definition to build for the worker robot
#define  MANAGER_ROBOT    1

#if defined(MANAGER_ROBOT)
#pragma message("building for the manager")
ai::robot_link       link( PORT5, "robot_32456_1", linkType::manager );
#else
#pragma message("building for the worker")
ai::robot_link       link( PORT13, "robot_32456_1", linkType::worker );
#endif

/*************************************

            axis defination

*************************************/
/*

robot_local : 

                  ^ y   Head  ^    ^--->
                  |           |    |<-- imu  
                  |
                  |
(rotation_center) |——————————> x


robot_global : 

                  ^ y
                  |           
                  |
                  |
         (middle) |——————————> x

*/

/**
 * Used to initialize code/tasks/devices added using tools in VEXcode Pro.
 * 
 * This should be called at the start of your int main function.
 */
void vexcodeInit( void ) {
  // nothing to initialize
}