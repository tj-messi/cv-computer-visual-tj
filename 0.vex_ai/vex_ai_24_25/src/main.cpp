/*----------------------------------------------------------------------------*/
/*                                                                            */
/*    Author:       TJU-CodeWeavers                                           */
/*    Created:      2023/11/1 23:12:20                                        */
/*    Description:  tjulib for V5 project                                     */
/*                                                                            */
/*----------------------------------------------------------------------------*/
#include "vex.h"
#include "tjulib.h"
#include <string>
// #include "ai_functions.h"
using namespace vex;
using namespace tjulib;

/*---------------  模式选择  ---------------*/
// 如果进行技能赛就def，否则注释，进行自动
//#define SKILL
// 如果用里程计就def，否则注释，用雷达
#define ODOM
// 如果要开启远程调试就def，否则就注释
#define Remotedeubug
const double car_width = 11.15;                   // 杞�璺�
const double r_wheel = 4 / 2;                     // 杞﹁疆鍗婂緞
const double gear_ratio = 0.667;                  // 搴曠洏鐢垫満-杞�鐨勯娇杞�浼犲姩姣旓紙鍔犻€熼厤姣斿氨澶т簬1锛屽噺閫熼厤姣斿氨灏忎簬1锛�
const double r_motor = r_wheel * gear_ratio ;     // 鐢垫満杞�瑙�-鐢垫満杞�鍛ㄧ殑鎹㈢畻姣�
const double cell  = 24;                          // 涓€涓�鍦板灚闀垮害(inches)
const double hOffset  = -5;                       // 閲岀▼璁″亸缃�锛坕nches锛�----浠庢棆杞�涓�蹇冨悜閲岀▼璁¤疆寤朵几鏂瑰悜浣滃瀭绾�
const double vOffset  = 5;                        // 閲岀▼璁″亸缃�锛坕nches锛�----浠庢棆杞�涓�蹇冨悜閲岀▼璁¤疆寤朵几鏂瑰悜浣滃瀭绾�
const double r_wheel_encoder = 2.75 / 2;          // 缂栫爜杞�鍛ㄩ暱
const double gps_offset_x = 0;                    // GPS鐨剎杞存柟鍚戝亸缃� 
const double gps_offset_y = 6.7;                  // GPS鐨剏杞存柟鍚戝亸缃� 
const double encoder_rotate_degree = 45;          // 缂栫爜杞�鏃嬭浆瑙掑害
/*************************************

            state flags

*************************************/
// gps鍧愭爣(甯︽棆杞�涓�蹇冨亸缃�淇�姝�)
double gps_x = 0;
double gps_y = 0;
double gps_heading = 0;
//vision鎺у埗
bool photoFlag = false;
bool abandon = true;
bool throwFlag = false;
bool reverseSpin = false;
bool forwardSpin = false;

bool ring_convey_spin = false;  // 鏄�鍚﹀紑濮嬭繘琛屽惛鐜�
int ring_color = 0;             // 瀵硅幏鍙栫殑鐜�鐨勯�滆壊杩涜�屾�€鏌ワ紝0鏄�娌℃湁鐜�锛�1鏄�钃濊壊鐜�锛�2鏄�绾㈣壊鐜�
 
/*************************************

            VEX devices

*************************************/
// A global instance of brain used for printing to the V5 Brain screen
brain  Brain;
// 搴曠洏鐢垫満 - 鍥涜�掑簳鐩�
motor L1 = motor(PORT13, ratio6_1, false);
motor L2 = motor(PORT13, ratio6_1, true);
motor L3 = motor(PORT13, ratio6_1, false);
motor L4 = motor(PORT13, ratio6_1, true);
motor R1 = motor(PORT13, ratio6_1, true);
motor R2 = motor(PORT13, ratio6_1, false);
motor R3 = motor(PORT13, ratio6_1, true);
motor R4 = motor(PORT13, ratio6_1, false); 
std::vector<vex::motor*> _leftMotors = {&L1, &L2, &L3,  &L4};
std::vector<vex::motor*> _rightMotors = {&R1, &R2, &R3, &R4};

// 搴曠洏鐢垫満 - 鍏�瑙掑簳鐩�
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

// 鎶�鍗囪噦
motor lift_armMotorA = motor(PORT8, ratio36_1, true);
motor lift_armMotorB = motor(PORT15, ratio36_1, false);
motor_group lift_arm = motor_group(lift_armMotorA, lift_armMotorB);

// 浼犻€佸甫
motor convey_beltMotorA = motor(PORT18, ratio36_1, true);
motor convey_beltMotorB = motor(PORT14, ratio36_1, false);
motor_group convey_belt = motor_group(convey_beltMotorA, convey_beltMotorB);

// 鍚哥悆
motor rollerMotorA = motor(PORT6, ratio18_1, true);
motor rollerMotorB = motor(PORT13, ratio18_1, false);
motor_group roller_group = motor_group(rollerMotorA, rollerMotorB);
// 閬ユ帶鍣�
controller Controller1 = controller(primary);
// 閫氫俊澶╃嚎
vex::message_link AllianceLink(PORT13, "tju1", linkType::worker);
// 閲岀▼璁�
encoder encoderHorizonal = encoder(Brain.ThreeWirePort.A);
encoder encoderVertical = encoder(Brain.ThreeWirePort.G);
// 瀵煎叆鏉�
motor side_bar = motor(PORT13, ratio18_1, false);
// imu鎯�鎬т紶鎰熷櫒
inertial imu = inertial(PORT17);  // 绗�浜屼釜鍙傛暟瑕佸啓right

// 姘斿姩浠�
pwm_out gas_push = pwm_out(Brain.ThreeWirePort.D);
pwm_out gas_lift = pwm_out(Brain.ThreeWirePort.E);
pwm_out gas_hold = pwm_out(Brain.ThreeWirePort.F);

// 璺濈�讳紶鎰熷櫒
distance DistanceSensor = distance(PORT13);
// gps
gps GPS_ = gps(PORT16, 0, 0, inches, 0);

// vision signature
vision::signature Red1 = vision::signature(3, 9051, 11375, 10213, -1977, -833, -1405, 1.6711680, 0);
vision::signature Red2 = vision::signature(3,  5461, 8761, 7111, -1457, -167, -812, 0.8144962, 0);
vision::signature Red3 = vision::signature(3,  8191, 9637, 8914, -1831, -735, -1283, 0.6828844, 0);

vision::signature Blue1 = vision::signature(2,  -4177, -3545, -3861, 6099, 7047, 6573, 1.2, 0);
vision::signature Blue2 = vision::signature(2, -5461, -3919,  -4690, 7721, 11045, 9383, 1.123142, 0);
vision::signature Blue3 = vision::signature(1,  -4177, -3545, -3861, 6099, 7047, 6573, 1.2, 0);
std::vector<vision::signature*>Red = {&Red1, &Red2, &Red3};
std::vector<vision::signature*>Blue = {&Blue1, &Blue2, &Blue3};
// vision
vision Vision = vision(PORT19, 50, Blue1, Blue2, Blue3);


// VEXcode generated functions
// define variable for remote controller enable/disable
bool RemoteControlCodeEnabled = true;

// manager and worker robots
// Comment out the following definition to build for the worker robot
#define  MANAGER_ROBOT    1

#if defined(MANAGER_ROBOT)
#pragma message("building for the manager")
ai::robot_link       link( PORT13, "robot_32456_1", linkType::manager );
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
(rotation_center) |鈥斺€斺€斺€斺€斺€斺€斺€斺€斺€�> x


robot_global : 

                  ^ y
                  |           
                  |
                  |
         (middle) |鈥斺€斺€斺€斺€斺€斺€斺€斺€斺€�> x

*/

/**
 * Used to initialize code/tasks/devices added using tools in VEXcode Pro.
 * 
 * This should be called at the start of your int main function.
 */
void vexcodeInit( void ) {
  // nothing to initialize
}

/**************************电机定义***********************************/
// ordinary chassis define
//std::vector<std::vector<vex::motor*>*> _chassisMotors = { &_leftMotors, &_rightMotors} ;
// oct chassis define
std::vector<std::vector<vex::motor*>*> _chassisMotors = {&_lfMotors, &_lbMotors, &_rfMotors, &_rbMotors};
/**************************调参区域***********************************/

// Definition of const variables
//const double PI = 3.1415926;

// imu零漂误差修正
double zero_drift_error = 0;  // 零漂误差修正，程序执行时不断增大
double correct_rate = 0.0000;

// 全局计时器
static timer global_time;  
// 竞赛模板类
competition Competition;
// vex-ai jeson nano comms
ai::jetson  jetson_comms;
static AI_RECORD local_map;
/*************************************

        pid configurations

*************************************/

/*configure meanings：
    ki, kp, kd, 
    integral's active zone (either inches or degrees), 
    error's thredhold      (either inches or degrees),
    minSpeed               (in voltage),
    stop_num               (int_type)
*/

pidParams   fwd_pid(10, 0.6, 0.3, 1, 2, 25, 15), 
            turn_pid(5, 0.05, 0.05, 3, 2, 20, 15), 
            cur_pid(8.0, 0.05, 0.15, 3, 1, 20, 15),
            straightline_pid(10, 0.1, 0.12, 5, 4, 1, 10),
            wheelmotor_pid(0.25, 0.01, 0.02, 50, 5, 0, 10);


/*************************************

        Instance for position

*************************************/
//Dif_Odom diff_odom(_leftMotors, _rightMotors,  PI * r_motor * 2, r_motor, imu);

// gps correction
tjulib::GPS gps_(GPS_, gps_offset_x, gps_offset_y);
// odom(of 45 degree) strategy
Odom odom(hOffset, vOffset, r_wheel_encoder, encoder_rotate_degree, encoderVertical, encoderHorizonal, imu);
// diff-odom strategy ----- diff_odom default
Context *PosTrack = new Context(&odom); 
// vector for all position strategy
std::vector<Position*>_PositionStrategy = {&odom};

/*************************************

        Instance for map

*************************************/
// local storage for latest data from the Jetson Nano
//AI_RECORD local_map;
HighStakeMap map(PosTrack->position);

/*************************************

        Instance for control

*************************************/

// ====Declaration of PID parameters and PID controllers ====
pidControl curControl(&cur_pid);
pidControl fwdControl(&fwd_pid);
pidControl turnControl(&turn_pid);
pidControl straightlineControl(&straightline_pid);
pidControl motorControl(&wheelmotor_pid);

// ====Declaration of Path Planner and Controller ====
// Declaration of rrt planner
RRT rrtPlanner(map.obstacleList, -72, 72, 2, 25, 10000, 12);
// Declaration of PurPursuit Controller
PurePursuit purepursuitControl(PosTrack->position);

// ====Declaration of Chassis Controller ====
// 底盘控制
//Ordi_SmartChassis FDrive(_chassisMotors, &motorControl, PosTrack->position, r_motor, &curControl, &fwdControl, &turnControl, car_width);
Oct_SmartChassis ODrive(_chassisMotors, &motorControl, PosTrack->position, r_motor, &curControl, &fwdControl, &turnControl, car_width, &purepursuitControl);

// ====Declaration of Upper Controller ====

/***************************
 
      thread define

 **************************/
// 远程调试
RemoteDebug remotedebug(PosTrack->position); 
// 远程调试
int RemoteDubug(){

#ifdef DashBoard
    remotedebug.PositionDebugSerial();
#else

#endif
    return 0;
}

/***************************
 
      initial pos set

 **************************/
// PosTrack 定位线程，在这里选择定位策略
int PositionTrack(){

    // _PositionStrategy has {&diff_odom, &odom}
    PosTrack = new Context(_PositionStrategy[0]);
    PosTrack->startPosition();
    return 0;

}

// 更新线程
int GPS_update(){
    
    timer time;
    time.clear();
    int flag = 1;
    while(1){
       
        imu.setHeading(GPS_.heading(deg), deg);

        gps_x = gps_.gpsX();
        gps_y = gps_.gpsY();
        gps_heading = GPS_.heading(deg);
        
        if((time.time(msec)-3000)<=10 && flag){
            imu.setHeading(GPS_.heading(deg), deg);
            // 第4秒的时候会更新一下坐标
            PosTrack->setPosition({gps_x, gps_y, GPS_.heading(deg) / 180 * 3.1415926535});
            
            printf("position initialization finish\n");

            flag = 0;
        }
        task::sleep(10);
        
    }
        
        
}
/***************************
 
    pre-autonomous run

 **************************/
// 设置初始位置、角度
#ifdef SKILL
    // 初始位置，单位为inches
    double init_pos_x = -59;
    double init_pos_y = 35.4;

    // 逆时针角度，范围在0 ~ 360°之间
    double initangle = 0;

#else
    // 初始位置，单位为inches
    double init_pos_x = 0;
    double init_pos_y = 0;

    // 逆时针角度，范围在0 ~ 360°之间
    double init_angle = 0;

#endif
void pre_auton(){
    thread PosTrack_(PositionTrack);
/***********是否开启远程调试************/
#ifdef Remotedeubug
    thread Remotedebug(RemoteDubug);
#endif
/***********imu、gps、distancesensor、vision等设备初始化************/  
    
    printf("pre-auton start\n");
    if(GPS_.installed()){
        GPS_.calibrate();
        while(GPS_.isCalibrating()) task::sleep(8);
        
    }

    // 这里考虑到只使用imu而不使用gps的情况
    if(imu.installed()){
        // 设置初始位置
        PosTrack->setPosition({init_pos_x, init_pos_y, init_angle});
    }
    
    if(GPS_.installed()){
        thread GPS_update_(GPS_update);
     }
    
    printf("pre-auton finish\n");
    task::sleep(3000);
}

/*********************************
 
    Dual-Communication Thread

 ***********************************/
static int received_flag = 0;
int sendTask(){

    while( !AllianceLink.isLinked() )
        this_thread::sleep_for(8);

    AllianceLink.send("run");
    Brain.Screen.print("successfully sended\n");
    return 0;
}
void confirm_SmallCar_Finished(const char* message, const char*linkname, double nums){

    received_flag = 1;
    Brain.Screen.print("successfully received\n");
}    
void demo_dualCommunication(){
    sendTask();  // 向联队车发送信息
    task::sleep(200);
    Brain.Screen.print("send thread jump out\n");
    while(1){
        AllianceLink.received("finished", confirm_SmallCar_Finished);
        task::sleep(200);
        if(received_flag){break;}
    }

}



void auto_Isolation(void) { 
  
  //test code
  while(1){
    if(local_map.detectionCount>0)
    {
      ODrive.simpleMove(50, 0, 0.4, 10);
    }
  }

}

void auto_Interaction(void) {

}
bool firstAutoFlag = true;

void autonomousMain(void) {

  if(firstAutoFlag)
    auto_Isolation();
  else 
    auto_Interaction();

  firstAutoFlag = false;
}

int main() {

   printf("get in /n");
  // local storage for latest data from the Jetson Nano
  static AI_RECORD local_map;

  // Run at about 15Hz
  int32_t loop_time = 33;

  // start the status update display
  thread t1(dashboardTask);

    // print through the controller to the terminal (vexos 1.0.12 is needed)
  // As USB is tied up with Jetson communications we cannot use
  // printf for debug.  If the controller is connected
  // then this can be used as a direct connection to USB on the controller
  // when using VEXcode.
  //
  //FILE *fp = fopen("/dev/serial2","wb");
   Competition.autonomous(autonomousMain);

  // Run the pre-autonomous function.
  //pre_auton();
  this_thread::sleep_for(loop_time);
 

  // Prevent main from exiting with an infinite loop.
  while (true) {
        // get last map data
      jetson_comms.get_data( &local_map );

      // set our location to be sent to partner robot
      link.set_remote_location( local_map.pos.x, local_map.pos.y, local_map.pos.az, local_map.pos.status );

      printf("%.2f %.2f %.2f\n", local_map.pos.x, local_map.pos.y, local_map.pos.az);

      // request new data    
      // NOTE: This request should only happen in a single task.    
      jetson_comms.request_map();

      this_thread::sleep_for(loop_time);
  }
}