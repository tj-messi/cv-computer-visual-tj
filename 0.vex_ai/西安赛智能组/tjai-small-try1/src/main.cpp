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
RRT rrtPlanner_short(map.obstacleList, -72, 72, 4, 25, 20000, 8);
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
int PositionTrack(){

    // _PositionStrategy has {&diff_odom, &odom}
    PosTrack = new Context(_PositionStrategy[0]);
    PosTrack->startPosition();
    return 0;

}

int sendTask(){

    while( !AllianceLink.isLinked() )
        this_thread::sleep_for(8);


    while(1){
        char *message = new(std::nothrow)char[256];
        sprintf(message, "%lf,%lf", GPS_.xPosition(inches), GPS_.yPosition(inches));
        AllianceLink.send(message);
        task::sleep(20);
        delete []message;
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
        
        if((time.time(msec)-3000) <= 50 && flag){
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
    thread CheckStuck_(CheckStuck);
    thread CheckRing_(CheckRing);
    thread send(sendTask);
    printf("pre-auton finish\n");
    task::sleep(3000);
}

/*********************************
 
    Dual-Communication Thread

 ***********************************/
static int received_flag = 0;


// Dual-Communication Demo



int go_out_roller(){
    while(1){
        manual = true;
        reinforce_stop = true;
        // 当进入桩内的时候退�?
        if((gps_x * gps_x + gps_y * gps_y) < 24){
            break;
        }
    }

    // 吸一下接着就停
    manual = true;
    ring_convey_spin = true;
    reinforce_stop = false;
    task::sleep(200);
    manual = true;
    reinforce_stop = true;

    // 等到走出来之后就套环
    while(1){
        // 当进入桩内的时候退�?
        if((gps_x * gps_x + gps_y * gps_y) >= 28){
            manual = true;
            ring_convey_spin = true;
            reinforce_stop = false;
            break;
        }
    }
    return 0;
}


struct oj_data
{
    double x;
    double y;
    int kind;
};
static std::vector<oj_data> my_map;
static double diff = 0.5;


/***************************
 
      autonomous run

 **************************/
void auto_Isolation(void) 
{
/*
    Point start_pt = {gps_x, gps_y};
    std::vector<Point> Path1 = rrtPlanner_short.optimal_rrt_planning(start_pt, (Point){48, 0, 0}, 4);  // 这里一定�?�强制类型转�?为Point
    ODrive.PathMove(Path1, 400, 100, 800000, 10, 1, 0);
*/

        while(1)
        {   
            {
                convey_belt.stop();
                roller_group.stop();
            }
            // 获取到最近的柱子
            int min_index;
            int min_distance = INT_MAX;
            for(int i = 0;i<my_map.size();i++){
                if(min_distance > sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y)) && my_map[i].kind == 0){
                    min_distance = sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y));
                    min_index = i;
                }
            }
            oj_data nearest_elem = my_map[min_index];
            // 停�?�旋�?
            ODrive.VRUN(0, 0, 0, 0);
            // 朝向�?
            ODrive.turnToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000,1,1);
            // 吃环
            ODrive.moveToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000);
            ODrive.simpleMove(80,180,1,10);
            gas_hold.state(100,pct);
           //往前走一段距�?
           task::sleep(800);
           ODrive.simpleMove(80,0,1,10);

           break;
        }
    while(1)
    {
            {
                convey_belt.spin(fwd,100,pct);
                roller_group.spin(fwd,-100,pct);
            }
            // ��ȡ������ĺ컷
            int min_index;
            int min_distance = INT_MAX;
            for(int i = 0;i<my_map.size();i++){
                if(min_distance > sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y)) && my_map[i].kind == 1){
                    min_distance = sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y));
                    min_index = i;
                }
            }
            oj_data nearest_elem = my_map[min_index];
            // ͣ???��??
            ODrive.VRUN(0, 0, 0, 0);

            //�Ի�
            ODrive.turnToTarget(Point{nearest_elem.x, nearest_elem.y}, 40, 2000);
            ODrive.moveToTarget(Point{nearest_elem.x, nearest_elem.y}, 40, 2000);
            task::sleep(800);
            //�ѻ���my_map??������ͼ
            my_map[min_index].x=1e5;
            my_map[min_index].y=1e5;
            task::sleep(800);
            ODrive.simpleMove(30,0,1,10);
    }

//     // 加分区坐�?
//     std::vector<Point> bonusAreas = {{60, 60}, {-60, 60}, {60, -60}, {-60, -60}};
//     // 固定桩坐�?
//     std::vector<Point> fixedStakes = {{60, 0}, {-60, 0}, {0, -60}, {0, 60}};

//     // 清前1/4�? 
//     // 取桩
//     std::vector<Point> Path1 = {{-48, 12}, {-34, 29}, {-28, 42}};
//     ODrive.turnToTarget((Point){-24, 48}, 40, 900, 1, 1);
//     ODrive.PathMove(Path1, 200, 100, 800000, 10, 1, 0);
//     ODrive.turnToTarget((Point){-24, 48}, 40, 700, 1, 1);
//     ODrive.moveToTarget((Point){-24, 48}, 70, 1100, 10);
//     gas_hold.state(100,pct);

//   //  ODrive.HSAct(1, (Point){-24, 48}, 80, 80, 8000, 10, 1, 0););
//     // 取环
//     ODrive.turnToTarget((Point){0, 48}, 40, 1100);
//     ODrive.moveToTarget((Point){-8,48}, 80, 1000, 10);

//     ODrive.HSAct(0, (Point){0, 48}, 80, 80, 1000, 15, 1, 0);

//     ODrive.moveToTarget((Point){-20,48}, 80, 1000, 10);
//     ODrive.turnToTarget((Point){2, 60}, 40, 1500);
//     ODrive.HSAct(0, (Point){2, 60}, 80, 80, 8000, 15, 1, 0);
//     ODrive.moveToTarget((Point){-39, 24}, 100, 800, 10);
//     ODrive.turnToTarget((Point){-48, 48}, 40, 1200);
//     ODrive.HSAct(0, (Point){-48, 48}, 1500, 80, 1500, 15, 1, 0);
     
//     manual = true;
//     ring_convey_spin = true;
//     reinforce_stop = false;
     

//     // ===吸左下�?�落===
//     // 吸�?�落�?
//      {
//      ODrive.moveToTarget((Point){-51, 50}, 40, 1800, 10);
//      ODrive.turnToAngle(-43, 50, 900);
//      ODrive.simpleMove(43, 0, 0.3, 10);

//      ODrive.turnToAngle(-45, 40, 0);
    
//     //再�?�转
//      ring_convey_spin = true;
//      reinforce_stop = false;


//     // ODrive.turnToAngle(0, 40, 450);
//      ODrive.simpleMove(70, 0, 0.8, 10);
//      task::sleep(150);
//      ODrive.turnToAngle(-20, 60, 350);
//      ODrive.simpleMove(40, 0, 0.3, 10);
//      ODrive.turnToAngle(0, 60, 350);
//      ODrive.simpleMove(40, 0, 0.3, 10);
//      task::sleep(550);

//      ODrive.turnToAngle(-45, 100, 350);

//      }
     
//     ODrive.HSAct(2, (Point)bonusAreas[1], 100, 80, 1500, 10, 1, 0);
    
//     //ODrive.HSAct(4, (Point){-48, 48}, 100, 80, 500, 10, 1, 0);


//      // 清后1/4�? 
//     // 跑场
//     ODrive.turnToAngle(90, 75, 1000);
//     ODrive.moveToTarget((Point){-28,48}, 70, 1400, 10);
//     ODrive.moveToTarget((Point){12,48}, 70, 1400, 10);

//     // 吃�??一�?半环
//     ODrive.HSAct(4, (Point){24, 48}, 75, 705, 1500, 10, 1, 0);
//     ODrive.turnToTarget((Point){24, 24}, 60, 1100, 1, 1);

//     // 持桩
//     ODrive.moveToTarget((Point){5, 48}, 40, 700, 1, 1);
//     ODrive.turnToTarget((Point){24, 24}, 40, 900, 1, 1);
//     ODrive.moveToTarget((Point){24, 24}, 70, 1100, 10);
//     gas_hold.state(100,pct);
//     ring_convey_spin = true;
//     reinforce_stop = false;
//     task::sleep(200);
//     //ODrive.HSAct(1, (Point){24, 24}, 60, 85, 500, 15, 1, 0);    
//     task::sleep(250);
//     ODrive.HSAct(0, (Point){48,24}, 60, 85, 1200, 10, 1, 0);
//     ODrive.moveToTarget((Point){20, 48}, 100, 500, 10);
//     ODrive.moveToTarget((Point){32, 57}, 100, 500, 10);
//     ODrive.HSAct(0, (Point){48, 48}, 60, 85, 700, 5, 1, 0);



//     ODrive.moveToTarget((Point){51, 50}, 40, 1200, 10);
//     ODrive.turnToAngle(-45, 75, 1000);
//     ring_convey_spin = false;
//     reinforce_stop = true;
//     ODrive.simpleMove(70, 180, 0.7, 10);
//     task::sleep(350);

   // ODrive.HSAct(2, (Point)bonusAreas[0], 100, 80, 1500, 10, 1, 0);



    // 动作空间:0取环, 1取桩, 2放桩, 3扣环, 4取半�? 





    // 套边�? 
   // ODrive.HSAct(3, (Point)fixedStakes[2], 75, 100, 1500, 10, 1, 0);
    //imu.setHeading(GPS_.heading(deg), deg);

   
    //thread go_out_roller(go_out_roller);
    //ODrive.HSAct(4, (Point){7, -7}, 60, 80, 1000, 10, 1, 0);
    //task::sleep(500);



}
/***************************
 
      skillautonomous run

 **************************/
void auto_Interaction(void) {

}
/***************************
 
      usercontrol run

 **************************/
bool firstAutoFlag = true;
void autonomousMain(void) {


  if(firstAutoFlag)
    auto_Isolation();
  else 
    auto_Interaction();

  firstAutoFlag = false;
}

void renew_map(void)
{
    jetson_comms.get_data( &local_map );
    for(int i=0;i<local_map.detectionCount;i++)
        {

            oj_data data;

            T data_x, data_y;
            if(1){

                data_x = local_map.detections[i].mapLocation.x * 39.3700788;
                data_y = -2+local_map.detections[i].mapLocation.y * 39.3700788;
            }else{  // jetson_nano��GPS���������⣬һֱ��(0, 0), ����Ҫ���ñ�����??????
                T local_data_x = local_map.detections[i].mapLocation.x * 39.3700788;
                T local_data_y = local_map.detections[i].mapLocation.y * 39.3700788;
                T sum_offset_x =  local_data_x;   
                T sum_offset_y = local_data_y;

                T theta = GPS_.heading(deg) / 180 * 3.1415926535;
                data_x = gps_x + ( sum_offset_y * sin(theta) + sum_offset_x * cos(theta) );
                data_y = gps_y + ( sum_offset_y * cos(theta) - sum_offset_x * sin(theta) );
            }

            //x y ���� : ��???��??��λ??+??����λ��
            data.x = data_x;
            data.y = data_y;
            // ���
            data.kind = local_map.detections[i].classID; 
            my_map.push_back(data);

        }
}

int main() {
    thread PosTrack_(PositionTrack);
  // local storage for latest data from the Jetson Nano
    printf("pre-auton start\n");
    if(GPS_.installed()){
        GPS_.calibrate();
        while(GPS_.isCalibrating()) task::sleep(8);
        
    }
    //thread receive(receivedTask);

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
  // Prevent main from exiting with an infinite loop.
  bool gps_jetson_nano_dead = false;
  while(1) {
    my_map = {};
    gps_jetson_nano_dead = false;
      // get last map data
      jetson_comms.get_data( &local_map );

        // ��Ҫ???jetson nano����GPS������һ??Ӧ��??????
        if(fabs(local_map.pos.x - 0) < 1e-6 && fabs(local_map.pos.y - 0) < 1e-6 && fabs(local_map.pos.rot - 0) < 1e-6){       // ��ȫ??(0, 0)??��������������²ſ��ܳ�??
            gps_jetson_nano_dead = true;

        }
        // ��֪�����ƶ�����Ԫ�صı����ڴ��??
        for(int i=0;i<local_map.detectionCount;i++)
        {

            oj_data data;

            T data_x, data_y;
            if(!gps_jetson_nano_dead){

                data_x = local_map.detections[i].mapLocation.x * 39.3700788;
                data_y = -2+local_map.detections[i].mapLocation.y * 39.3700788;
            }else{  // jetson_nano��GPS���������⣬һֱ��(0, 0), ����Ҫ���ñ�����??????
                T local_data_x = local_map.detections[i].mapLocation.x * 39.3700788;
                T local_data_y = local_map.detections[i].mapLocation.y * 39.3700788;
                T sum_offset_x =  local_data_x;   
                T sum_offset_y = local_data_y;

                T theta = GPS_.heading(deg) / 180 * 3.1415926535;
                data_x = gps_x + ( sum_offset_y * sin(theta) + sum_offset_x * cos(theta) );
                data_y = gps_y + ( sum_offset_y * cos(theta) - sum_offset_x * sin(theta) );
            }

            //x y ���� : ��???��??��λ??+??����λ��
            data.x = data_x;
            data.y = data_y;
            // ���
            data.kind = local_map.detections[i].classID; 
            my_map.push_back(data);

        }
                // ����ʱͨ����������????
          if(Controller1.ButtonX.pressing()){ 
             autonomousMain();
             thread renew_map(renew_map);
         }

#ifdef JETSON_NANO_VISION_DEBUG
        // jetson_nano��֪����
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

