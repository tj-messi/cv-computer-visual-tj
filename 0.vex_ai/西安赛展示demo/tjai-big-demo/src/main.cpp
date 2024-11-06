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
// 如果�?红方就def，否则就注释
#define RED
// 如果开启jetson_nano感知调试就def，否则就注释
#define JETSON_NANO_VISION_DEBUG

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
// 红方标志
bool is_red = true;
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

pidParams   fwd_pid(8, 0.3, 0.3, 2, 2, 12, 15), 
            turn_pid(4, 0.15, 0.15, 45, 1, 10, 15), 
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
RRT rrtPlanner_short(map.obstacleList, -72, 72, 2, 25, 20000, 4);
RRT rrtPlanner_long(map.obstacleList, -72, 72, 3, 20, 20000, 12);
// Declaration of PurPursuit Controller
PurePursuit purepursuitControl(PosTrack->position);

// ====Declaration of Chassis Controller ====
// 底盘控制
//Ordi_SmartChassis FDrive(_chassisMotors, &motorControl, PosTrack->position, r_motor, &curControl, &fwdControl, &turnControl, car_width);
Oct_SmartChassis ODrive(_chassisMotors, &motorControl, PosTrack->position, r_motor, &curControl, &fwdControl, &turnControl, car_width, &purepursuitControl, &map, &rrtPlanner_short, &rrtPlanner_long);



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

double hx=0, hy=0, zx=0, zy=0;
int siz = 0;
int PositionTrack(){

    // _PositionStrategy has {&diff_odom, &odom}
    PosTrack = new Context(_PositionStrategy[0]);
    PosTrack->startPosition();
    return 0;

}
void Get_SmallCar_GPS(const char* message, const char*linkname, double nums){
    
   sscanf(message, "%d,%lf,%lf,%lf,%lf",&siz, &hx, &hy, &zx, &zy);
    printf("my.size: %d, hx: %lf, hy: %lf,zx: %lf,zy: %lf\n", siz, hx, hy, zx, zy);

    printf("successfully received\n");
}   
int receivedTask(){

    while( !AllianceLink.isLinked() )
        {
            this_thread::sleep_for(8);
            printf("waiting for link\n");
        }
    
    gps_x_small = 0;
    gps_y_small = 0;
    while(1){
        AllianceLink.received(Get_SmallCar_GPS);
        task::sleep(20);

    }
    return 0;
}
  
// 更新线程
int GPS_update(){
    
    timer time;
    time.clear();
    int flag = 1;
    imu.setHeading(GPS_.heading(deg), deg);
    while(1){
       
       // imu.setHeading(GPS_.heading(deg), deg);

        gps_x = gps_.gpsX();
        gps_y = gps_.gpsY();
        gps_heading = GPS_.heading(deg);
        
        if((time.time(msec)-3000)<=50 && flag){
            imu.setHeading(GPS_.heading(deg), deg);
            imu.setRotation(GPS_.heading(deg), deg);
            // �?4秒的时候会更新一下坐�?
            PosTrack->setPosition({gps_x, gps_y, GPS_.heading(deg) / 180 * 3.1415926535});
            
            printf("position initialization finish\n");

            flag = 0;
        }
        task::sleep(10);
         
    }
        
        
}               
 void VisionTest(){
    while(1){
        Vision_front.takeSnapshot(Stake_Red);
        if(Vision_front.largestObject.exists){
            int x = Vision_front.largestObject.centerX;
            printf("Red : center_x : %d\n", x);
        }
        Vision_front.takeSnapshot(Stake_Blue);
        if(Vision_front.largestObject.exists){
            int x = Vision_front.largestObject.centerX;
            printf("Blue : center_x : %d\n", x);
        }
        Vision_front.takeSnapshot(Stake_Yellow);
        if(Vision_front.largestObject.exists){
            int x = Vision_front.largestObject.centerX;
            printf("Yellow : center_x : %d\n", x);
        }
        task::sleep(30);
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
#ifdef RED
    is_red = true;
#else
    is_red = false;
#endif
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
    thread receive(receivedTask);

    // 这里考虑到只使用imu而不使用gps的情�?
    if(imu.installed()){
        // 设置初�?�位�?
        PosTrack->setPosition({init_pos_x, init_pos_y, init_angle});
    }
    // GPS更新线程
    if(GPS_.installed()){
        thread GPS_update_(GPS_update);
    }
    //thread testvision(VisionTest);
    // 吸环线程
    thread GetRing_(GetRing);
    //thread CheckStuck_(CheckStuck);
    thread CheckRing_(CheckRing);

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


struct oj_data
{
    double x;
    double y;
    int kind;
};
static std::vector<oj_data> my_map;
static double diff = 0.5;


void auto_Isolation(void) 
{

    {                
        //     zy+=-8;
        //     zx+=-16;
        //     ODrive.VRUN(0, 0, 0, 0);
        //     // 朝向�?
        //     ODrive.turnToTarget(Point{zy, zx}, 80, 2000,1,1);
        //     // 吃环
        //     ODrive.moveToTarget(Point{zy, zx}, 80, 12000);

        //     {
        //         // if(((zy-gps_y)*(zy-gps_y)+(zx-gps_x)*(zx-gps_x))<=500)
        //         // {
        //         //     ODrive.setStop(hold);
        //         // }
        //         //ODrive.turnToTarget(Point{zy, zx}, 80, 2000,1,1);
        //     }

        //     ODrive.simpleMove(80,180,0.610);
        //     gas_hold.state(100,pct);
        //    //往前走一段距�?
        //    task::sleep(800);
        //    ODrive.simpleMove(80,0,0.6,10);
    }

    {   
        // while(1)
        // {   
        //     {
        //         convey_belt.stop();
        //         roller_group.stop();
        //     }
        //     // 获取到最近的柱子
        //     int min_index;
        //     int min_distance = INT_MAX;
        //     for(int i = 0;i<my_map.size();i++){
        //         if(min_distance > sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y)) && my_map[i].kind == 0){
        //             min_distance = sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y));
        //             min_index = i;
        //         }
        //     }
        //     oj_data nearest_elem = my_map[min_index];
        //     // 停�?�旋�?
        //     ODrive.VRUN(0, 0, 0, 0);
        //     // 朝向�?
        //     ODrive.turnToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000,1,1);
        //     // 吃环
        //     ODrive.moveToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000);
        //     ODrive.simpleMove(80,180,1,10);
        //     gas_hold.state(100,pct);
        //    //往前走一段距�?
        //    task::sleep(800);
        //    ODrive.simpleMove(80,0,1,10);

        //    break;
    }
        
    while(1)
    {
            {
                convey_belt.spin(fwd,100,pct);
                roller_group.spin(fwd,-100,pct);
            }
            // 获取到最近的红环
            int min_index;
            int min_distance = INT_MAX;
            for(int i = 0;i<my_map.size();i++){
                if(min_distance > sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y)) && my_map[i].kind == 1){
                    min_distance = sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y));
                    min_index = i;
                }
            }
            oj_data nearest_elem = my_map[min_index];
            // 停�?�旋�?
            ODrive.VRUN(0, 0, 0, 0);

            //吃环
            ODrive.turnToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000);
            ODrive.moveToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000);
            task::sleep(800);
            //把环从my_map�?丢出地图
            nearest_elem.x=1e5;
            nearest_elem.y=1e5;
            task::sleep(800);
            ODrive.simpleMove(80,0,1,10);
    }
}

void auto_Interaction(void) {

}
// �?动模�? 先ioslation 后interaction
bool firstAutoFlag = true;
void autonomousMain(void) {


  if(firstAutoFlag)
    auto_Isolation();
  else 
    auto_Interaction();

  firstAutoFlag = false;
}




int main() {

    thread PosTrack_(PositionTrack);

/***********imu、gps、distancesensor、vision等�?��?�初始化************/  
    
    printf("pre-auton start\n");
    if(GPS_.installed()){
        GPS_.calibrate();
        while(GPS_.isCalibrating()) task::sleep(8);
        
    }
    thread receive(receivedTask);

    // 这里考虑到只使用imu而不使用gps的情�?
    if(imu.installed()){
        // 设置初�?�位�?
        PosTrack->setPosition({init_pos_x, init_pos_y, init_angle});
    }
    // GPS更新线程
    if(GPS_.installed()){
        thread GPS_update_(GPS_update);
    }
    //thread testvision(VisionTest);
    // 吸环线程
    thread GetRing_(GetRing);
    //thread CheckStuck_(CheckStuck);
    thread CheckRing_(CheckRing);

    printf("pre-auton finish\n");
    
  task::sleep(4000);
  // local storage for latest data from the Jetson Nano
  static AI_RECORD local_map;

  // Run at about 15Hz
  int32_t loop_time = 33;

  // start the status update display
  thread t1(dashboardTask);

  {
    //thread t2(receivedTask);
  }

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


  bool gps_jetson_nano_dead = false;
  while(1) {
    my_map = {};
    gps_jetson_nano_dead = false;
      // get last map data
      jetson_comms.get_data( &local_map );

        // 需要�?�jetson nano处的GPS死掉做一�?应急�?��??
        if(fabs(local_map.pos.x - 0) < 1e-6 && fabs(local_map.pos.y - 0) < 1e-6 && fabs(local_map.pos.rot - 0) < 1e-6){       // 完全�?(0, 0)�?有在死掉的情况下才可能出�?
            gps_jetson_nano_dead = true;

        }
        // 感知到的移动场地元素的本地内存存�?
        for(int i=0;i<local_map.detectionCount;i++)
        {

            oj_data data;

            T data_x, data_y;
            if(!gps_jetson_nano_dead){

                data_x = local_map.detections[i].mapLocation.x * 39.3700788;
                data_y = -2+local_map.detections[i].mapLocation.y * 39.3700788;
            }else{  // jetson_nano读GPS出现了问题，一直是(0, 0), 则需要利用本地信�?�?�?
                T local_data_x = local_map.detections[i].mapLocation.x * 39.3700788;
                T local_data_y = local_map.detections[i].mapLocation.y * 39.3700788;
                T sum_offset_x = camera_offset_x + local_data_x;   
                T sum_offset_y = camera_offset_y + local_data_y;

                T theta = GPS_.heading(deg) / 180 * 3.1415926535;
                data_x = gps_x + ( sum_offset_y * sin(theta) + sum_offset_x * cos(theta) );
                data_y = gps_y + ( sum_offset_y * cos(theta) - sum_offset_x * sin(theta) );
            }

            //x y 坐标 : 相�?�于�?己位�?+�?己的位置
            data.x = data_x;
            data.y = data_y;
            // 类别
            data.kind = local_map.detections[i].classID; 
            my_map.push_back(data);

        }
                // 调试时通过按键进入�?�?
         if(Controller1.ButtonX.pressing()){ 
             autonomousMain();
         }

#ifdef JETSON_NANO_VISION_DEBUG
        // jetson_nano感知调试
        if(my_map.size()>0){
            convey_belt.spin(fwd,100,pct);
            roller_group.spin(fwd,-100,pct);
        }
        else{
            convey_belt.stop();
            roller_group.stop();
        }
#endif

      // set our location to be sent to partner robot
      link.set_remote_location( local_map.pos.x, local_map.pos.y, local_map.pos.az, local_map.pos.status );

      // fprintf(fp, "%.2f %.2f %.2f\n", local_map.pos.x, local_map.pos.y, local_map.pos.az)

      // request new data    
      // NOTE: This request should only happen in a single task.    
      jetson_comms.request_map();

      // Allow other tasks to run
      this_thread::sleep_for(loop_time);
  }
}