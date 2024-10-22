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

using namespace vex;
using namespace tjulib;

/*---------------  模式选择  ---------------*/
// 如果进�?�技能赛就def，否则注释，进�?�自�?
//#define SKILL
// 如果用里程�?�就def，否则注释，用雷�?
#define ODOM
// 如果要开�?远程调试就def，否则就注释
#define Remotedeubug


/**************************电机定义***********************************/
// ordinary chassis define
//std::vector<std::vector<vex::motor*>*> _chassisMotors = { &_leftMotors, &_rightMotors} ;
// oct chassis define
std::vector<std::vector<vex::motor*>*> _chassisMotors = {&_lfMotors, &_lbMotors, &_rfMotors, &_rbMotors};
/**************************调参区域***********************************/

// Definition of const variables
//const double PI = 3.1415926;

// imu零漂�?�?�?�?
double zero_drift_error = 0;  // 零漂�?�?�?正，程序执�?�时不断增大
double correct_rate = 0.0000;

// 全局计时�?
static timer global_time;  
// 竞赛模板�?
competition Competition;
// vex-ai jeson nano comms
ai::jetson  jetson_comms;

/*************************************

        pid configurations

*************************************/

/*configure meanings�?
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
AI_RECORD local_map;
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
            // �?4秒的时候会更新一下坐�?
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
// 设置初�?�位�?、�?�度
#ifdef SKILL
    // 初�?�位�?，单位为inches
    double init_pos_x = -59;
    double init_pos_y = 35.4;

    // 逆时针�?�度，范围在0 ~ 360°之间
    double initangle = 0;

#else
    // 初�?�位�?，单位为inches
    double init_pos_x = 0;
    double init_pos_y = 0;

    // 逆时针�?�度，范围在0 ~ 360°之间
    double init_angle = 0;

#endif
void pre_auton(){
    thread PosTrack_(PositionTrack);
/***********�?否开�?远程调试************/
#ifdef Remotedeubug
    thread Remotedebug(RemoteDubug);
#endif
/***********imu、gps、distancesensor、vision等�?��?�初始化************/  
    
    printf("pre-auton start\n");
    if(GPS_.installed()){
        GPS_.calibrate();
        while(GPS_.isCalibrating()) task::sleep(8);
        
    }

    // 这里考虑到只使用imu而不使用gps的情�?
    if(imu.installed()){
        // 设置初�?�位�?
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
// Dual-Communication Demo
void demo_dualCommunication(){
    sendTask();  // 向联队车发送信�?
    task::sleep(200);
    Brain.Screen.print("send thread jump out\n");

    /************************
      
      发送完信号后执行的程序
      
    ************************/

    // 等待一�?
    while(1){
        AllianceLink.received("finished", confirm_SmallCar_Finished);
        task::sleep(200);
        if(received_flag){break;}
    }

}

void auto_Isolation(void) {
    while(1)
    {
        if(local_map.detectionCount > 0)
            ODrive.simpleMove(50,0,0.5,10);
    }

}
void auto_Interaction(void) {
    //ODrive.simpleMove(50,0,0.5,10);
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

  // local storage for latest data from the Jetson Nano
  static AI_RECORD local_map;

  // Run at about 15Hz
  int32_t loop_time = 33;

  // start the status update display
  thread t1(dashboardTask);

  // Set up callbacks for autonomous and driver control periods.
  Competition.autonomous(autonomousMain);

  // print through the controller to the terminal (vexos 1.0.12 is needed)
  // As USB is tied up with Jetson communications we cannot use
  // printf for debug.  If the controller is connected
  // then this can be used as a direct connection to USB on the controller
  // when using VEXcode.
  //
  //FILE *fp = fopen("/dev/serial2","wb");
  this_thread::sleep_for(loop_time);

  //Arm.setVelocity(60, percent);

  while(1) {
       printf("in loop \n");
      // get last map data
      jetson_comms.get_data( &local_map );

      // set our location to be sent to partner robot
      link.set_remote_location( local_map.pos.x, local_map.pos.y, local_map.pos.az, local_map.pos.status );

      //fprintf(fp, "%.2f %.2f %.2f\n", local_map.pos.x, local_map.pos.y, local_map.pos.az);

      // request new data    
      // NOTE: This request should only happen in a single task.    
      jetson_comms.request_map();

      // Allow other tasks to run
      this_thread::sleep_for(loop_time);
  }
}