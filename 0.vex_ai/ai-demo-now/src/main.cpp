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

/*---------------  �?″紡�?夋�??  ---------------*/
// 濡傛灉杩涜�屾妧鑳借�?�灏眃ef锛屽惁鍒欐敞閲�?�紝杩涜�岃嚜鍔�?
//#define SKILL
// 濡傛灉鐢ㄩ噷绋�??�″氨def锛屽惁鍒欐敞閲�?�紝�?ㄩ浄杈�
#define ODOM
// 濡傛灉瑕佸紑鍚�杩滅▼璋�?�?灏眃ef锛屽惁鍒欏氨娉ㄩ�?
#define Remotedeubug
// 濡傛灉鏄�绾㈡柟灏眃ef锛屽惁鍒欏氨娉ㄩ�?
#define RED
// 濡傛灉�??�?鍚痡etson_nano鎰熺煡璋�?�?灏眃ef锛屽惁鍒欏氨娉ㄩ�?
//#define JETSON_NANO_VISION_DEBUG

/**************************鐢垫満瀹氫�?***********************************/
// ordinary chassis define
//std::vector<std::vector<vex::motor*>*> _chassisMotors = { &_leftMotors, &_rightMotors} ;
// oct chassis define
std::vector<std::vector<vex::motor*>*> _chassisMotors = {&_lfMotors, &_lbMotors, &_rfMotors, &_rbMotors};
/**************************璋冨�?鍖哄�?***********************************/

// Definition of const variables
//const double PI = 3.1415926;

// imu闆舵紓璇��?��淇�姝�
double zero_drift_error = 0;  // 闆舵紓璇��?��淇�姝ｏ紝绋�?�?鎵ц�屾�?�涓嶆柇澧炲�?
double correct_rate = 0.0000;

// 鍏ㄥ�?璁℃椂鍣�?
static timer global_time;  
// 绔炶禌妯℃澘�?�?
competition Competition;
// vex-ai jeson nano comms
ai::jetson  jetson_comms;
// 绾㈡柟鏍囧織
bool is_red = true;
/*************************************

        pid configurations

*************************************/

/*configure meanings锛�
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
RRT rrtPlanner(map.obstacleList, -72, 72, 3, 25, 20000, 12);
// Declaration of PurPursuit Controller
PurePursuit purepursuitControl(PosTrack->position);

// ====Declaration of Chassis Controller ====
// 搴曠洏鎺у�?
//Ordi_SmartChassis FDrive(_chassisMotors, &motorControl, PosTrack->position, r_motor, &curControl, &fwdControl, &turnControl, car_width);
Oct_SmartChassis ODrive(_chassisMotors, &motorControl, PosTrack->position, r_motor, &curControl, &fwdControl, &turnControl, car_width, &purepursuitControl, &map, &rrtPlanner);



/***************************
 
      thread define

 **************************/
// 杩滅▼璋�?�?
RemoteDebug remotedebug(PosTrack->position); 
// 杩滅▼璋�?�?
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
// PosTrack 瀹氫綅绾跨▼锛屽�?杩欓噷閫夋�?�瀹氫綅绛�?�?
int PositionTrack(){

    // _PositionStrategy has {&diff_odom, &odom}
    PosTrack = new Context(_PositionStrategy[0]);
    PosTrack->startPosition();
    return 0;

}
void Get_SmallCar_GPS(const char* message, const char*linkname, double nums){
    
   sscanf(message, "%lf,%lf", &gps_x_small, &gps_y_small);
    printf("%lf,%lf\n", gps_x_small, gps_y_small);

    Brain.Screen.print("successfully received\n");
}   
int receivedTask(){

    while( !AllianceLink.isLinked() )
        this_thread::sleep_for(8);
    
    gps_x_small = 0;
    gps_y_small = 0;
    while(1){
        AllianceLink.received(Get_SmallCar_GPS);
        task::sleep(20);

    }
    return 0;
}
  
// 鏇存柊绾跨▼
int GPS_update(){
    
    timer time;
    time.clear();
    int flag = 1;
    imu.setHeading(GPS_.heading(deg), deg);
    while(1){
       
      //  imu.setHeading(GPS_.heading(deg), deg);

        gps_x = gps_.gpsX();
        gps_y = gps_.gpsY();
        gps_heading = GPS_.heading(deg);
        
        if((time.time(msec)-3000)<=50 && flag){
            imu.setHeading(GPS_.heading(deg), deg);
            imu.setRotation(GPS_.heading(deg), deg);
            // 绗�4绉掔殑鏃跺€欎細鏇存柊涓�?涓�??潗鏍�?
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
// 璁剧疆鍒濆�嬩綅缃�銆佽�掑�?
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
    sendTask();  // 鍚戣仈闃熻溅鍙戦�?佷俊�?�?
    task::sleep(200);
    Brain.Screen.print("send thread jump out\n");

    /************************
      
      鍙戦�?佸畬淇″彿鍚庢墽琛岀殑绋�?�?
      
    ************************/

    // 绛�?�緟涓€涓�
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


void auto_Isolation(void) {
    while(1)
    {   
        if(my_map.size()>0)
        {   

            // 鑾峰彇鍒版渶杩戠殑鐜�?
            int min_index;
            int min_distance = INT_MAX;
            for(int i = 0;i<my_map.size();i++){
                if(min_distance > sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y))){
                    min_distance = sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y));
                    min_index = i;
                }
            }
            oj_data nearest_elem = my_map[min_index];
            // 鍋滄�㈡棆杞�?
            ODrive.VRUN(0, 0, 0, 0);
            // 鏈濆悜鐜�?
            roller_group.spin(forward, 70, pct);
            ODrive.turnToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000);
            ODrive.moveToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000);
           // 鍚冪�?
           //ODrive.HSAct(0, Point{nearest_elem.x, nearest_elem.y}, 100, 100, 5000, 20);
           task::sleep(800);
            
        }else{
            ODrive.VRUN(35, 35, 35, 35);
        }
    }
}
void auto_Interaction(void) {

}
// 鑷�鍔ㄦā�?�? 鍏坕oslation 鍚巌nteraction
bool firstAutoFlag = true;
void autonomousMain(void) {


  if(firstAutoFlag)
    auto_Isolation();
  else 
    auto_Interaction();

  firstAutoFlag = false;
}




int main() {


    if(GPS_.installed()){
        GPS_.calibrate();
        while(GPS_.isCalibrating()) task::sleep(8);
        thread gps_update(GPS_update);
    }

    // 杩欓噷鑰�?檻鍒板彧浣跨�?imu鑰屼笉浣跨敤gps鐨勬儏鍐�?
    // if(imu.installed()){
    //     // 璁剧疆鍒濆�嬩綅缃�
    //     PosTrack->setPosition({init_pos_x, init_pos_y, init_angle});
    // }
  task::sleep(4000);
  // local storage for latest data from thSe Jetson Nano
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

  bool gps_jetson_nano_dead = false;
  while(1) {
    my_map = {};
    gps_jetson_nano_dead = false;
      // get last map data
      jetson_comms.get_data( &local_map );

        // 闇€瑕佸��?�etson nano澶勭殑GPS姝绘帀鍋氫竴涓�搴旀�?ラ��?��
        if(fabs(local_map.pos.x - 0) < 1e-6 && fabs(local_map.pos.y - 0) < 1e-6 && fabs(local_map.pos.rot - 0) < 1e-6){       // 瀹屽叏鐨�?(0, 0)鍙�鏈�?�湪姝绘帀鐨勬儏鍐典笅鎵嶅彲鑳藉嚭鐜�
            gps_jetson_nano_dead = true;

        }
        // 鎰熺煡鍒�?殑绉诲姩鍦哄湴鍏�?礌鐨�?湰鍦板唴瀛樺瓨鍌�?
        for(int i=0;i<local_map.detectionCount;i++)
        {

            oj_data data;

            T data_x, data_y;
            if(!gps_jetson_nano_dead)
            {
                data_x = local_map.detections[i].mapLocation.x * 39.3700788;
                data_y = local_map.detections[i].mapLocation.y * 39.3700788;
            }
            else
            {  // jetson_nano璇籊PS鍑虹幇浜嗛棶棰橈紝涓�?鐩存�?(0, 0), 鍒欓渶瑕佸埄�?ㄦ湰鍦�?�俊�?�淇��?��
                T local_data_x = local_map.detections[i].mapLocation.x * 39.3700788;
                T local_data_y = local_map.detections[i].mapLocation.y * 39.3700788;
                T sum_offset_x = camera_offset_x + local_data_x;   
                T sum_offset_y = camera_offset_y + local_data_y;
                
                T theta = GPS_.heading(deg) / 180 * 3.1415926535;
                data_x = gps_x + ( sum_offset_y * sin(theta) + sum_offset_x * cos(theta) );
                data_y = gps_y + ( sum_offset_y * cos(theta) - sum_offset_x * sin(theta) );
            }

            //x y 鍧愭�? : 鐩�?��逛簬鑷�宸变綅缃�?+鑷�宸辩殑浣嶇疆
            data.x = data_x;
            data.y = data_y;
            // �?诲埆
            data.kind = local_map.detections[i].classID; 
            my_map.push_back(data);

        }
                // 璋冭�?鏃堕�?氳繃鎸�?�敭杩涘叆鑷�鍔�?
         if(Controller1.ButtonX.pressing()){ 
             autonomousMain();
         }

#ifdef JETSON_NANO_VISION_DEBUG
        // jetson_nano鎰熺煡璋�?�?
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