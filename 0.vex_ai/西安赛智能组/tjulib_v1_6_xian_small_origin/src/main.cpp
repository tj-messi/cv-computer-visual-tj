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

/*---------------  æ¨¡å¼é€‰æ‹©  ---------------*/
// å¦‚æœè¿›è?ŒæŠ€èƒ½èµ›å°±defï¼Œå¦åˆ™æ³¨é‡Šï¼Œè¿›è?Œè‡ªåŠ?
//#define SKILL
// å¦‚æœç”¨é‡Œç¨‹è?¡å°±defï¼Œå¦åˆ™æ³¨é‡Šï¼Œç”¨é›·è¾?
#define ODOM
// å¦‚æœè¦å¼€å?è¿œç¨‹è°ƒè¯•å°±defï¼Œå¦åˆ™å°±æ³¨é‡Š
#define Remotedeubug
// å¦‚æœæ˜?çº¢æ–¹å°±defï¼Œå¦åˆ™å°±æ³¨é‡Š
#define RED

/**************************ç”µæœºå®šä¹‰***********************************/
// ordinary chassis define
//std::vector<std::vector<vex::motor*>*> _chassisMotors = { &_leftMotors, &_rightMotors} ;
// oct chassis define
std::vector<std::vector<vex::motor*>*> _chassisMotors = {&_lfMotors, &_lbMotors, &_rfMotors, &_rbMotors};
/**************************è°ƒå‚åŒºåŸŸ***********************************/

// Definition of const variables
//const double PI = 3.1415926;

// imué›¶æ¼‚è¯?å·?ä¿?æ­?
double zero_drift_error = 0;  // é›¶æ¼‚è¯?å·?ä¿?æ­£ï¼Œç¨‹åºæ‰§è?Œæ—¶ä¸æ–­å¢å¤§
double correct_rate = 0.0000;

// å…¨å±€è®¡æ—¶å™?
static timer global_time;  
// ç«èµ›æ¨¡æ¿ç±?
competition Competition;
// vex-ai jeson nano comms
ai::jetson  jetson_comms;
// çº¢æ–¹æ ‡å¿—
bool is_red = true;
/*************************************

        pid configurations

*************************************/

/*configure meaningsï¼?
    ki, kp, kd, 
    integral's active zone (either inches or degrees), 
    error's thredhold      (either inches or degrees),
    minSpeed               (in voltage),
    stop_num               (int_type)
*/

pidParams   fwd_pid(2.7, 0.08, 0.08, 3, 3, 5, 15), 
            turn_pid(0.8, 0.08, 0.05, 45, 1, 10, 15), 
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
// åº•ç›˜æ§åˆ¶
//Ordi_SmartChassis FDrive(_chassisMotors, &motorControl, PosTrack->position, r_motor, &curControl, &fwdControl, &turnControl, car_width);
Oct_SmartChassis ODrive(_chassisMotors, &motorControl, PosTrack->position, r_motor, &curControl, &fwdControl, &turnControl, car_width, &purepursuitControl, &map, &rrtPlanner_short, &rrtPlanner_long);



/***************************
 
      thread define

 **************************/
// è¿œç¨‹è°ƒè¯•
RemoteDebug remotedebug(PosTrack->position); 
// è¿œç¨‹è°ƒè¯•
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
// PosTrack å®šä½çº¿ç¨‹ï¼Œåœ¨è¿™é‡Œé€‰æ‹©å®šä½ç­–ç•¥
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
  
// æ›´æ–°çº¿ç¨‹
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
            // ç¬?4ç§’çš„æ—¶å€™ä¼šæ›´æ–°ä¸€ä¸‹åæ ?
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
// è®¾ç½®åˆå?‹ä½ç½?ã€è?’åº¦
#ifdef SKILL
    // åˆå?‹ä½ç½?ï¼Œå•ä½ä¸ºinches
    double init_pos_x = -59;
    double init_pos_y = 35.4;

    // é€†æ—¶é’ˆè?’åº¦ï¼ŒèŒƒå›´åœ¨0 ~ 360Â°ä¹‹é—´
    double initangle = 0;

#else
    // åˆå?‹ä½ç½?ï¼Œå•ä½ä¸ºinches
    double init_pos_x = 0;
    double init_pos_y = 0;

    // é€†æ—¶é’ˆè?’åº¦ï¼ŒèŒƒå›´åœ¨0 ~ 360Â°ä¹‹é—´
    double init_angle = 0;

#endif
void pre_auton(){
#ifdef RED
    is_red = true;
#else
    is_red = false;
#endif
    thread PosTrack_(PositionTrack);
/***********æ˜?å¦å¼€å?è¿œç¨‹è°ƒè¯•************/
#ifdef Remotedeubug
    thread Remotedebug(RemoteDubug);
#endif
/***********imuã€gpsã€distancesensorã€visionç­‰è?¾å?‡åˆå§‹åŒ–************/  
    
    printf("pre-auton start\n");
    if(GPS_.installed()){
        GPS_.calibrate();
        while(GPS_.isCalibrating()) task::sleep(8);
        
    }


    // è¿™é‡Œè€ƒè™‘åˆ°åªä½¿ç”¨imuè€Œä¸ä½¿ç”¨gpsçš„æƒ…å†?
    if(imu.installed()){
        // è®¾ç½®åˆå?‹ä½ç½?
        PosTrack->setPosition({init_pos_x, init_pos_y, init_angle});
    }
    // GPSæ›´æ–°çº¿ç¨‹
    if(GPS_.installed()){
        thread GPS_update_(GPS_update);
    }
    //thread testvision(VisionTest);
    // å¸ç¯çº¿ç¨‹
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
        // å½“è¿›å…¥æ¡©å†…çš„æ—¶å€™é€€å‡?
        if((gps_x * gps_x + gps_y * gps_y) < 24){
            break;
        }
    }

    // å¸ä¸€ä¸‹æ¥ç€å°±åœ
    manual = true;
    ring_convey_spin = true;
    reinforce_stop = false;
    task::sleep(200);
    manual = true;
    reinforce_stop = true;

    // ç­‰åˆ°èµ°å‡ºæ¥ä¹‹åå°±å¥—ç¯
    while(1){
        // å½“è¿›å…¥æ¡©å†…çš„æ—¶å€™é€€å‡?
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
void autonomous(){
/*
    Point start_pt = {gps_x, gps_y};
    std::vector<Point> Path1 = rrtPlanner_short.optimal_rrt_planning(start_pt, (Point){48, 0, 0}, 4);  // è¿™é‡Œä¸€å®šè?å¼ºåˆ¶ç±»å‹è½¬æ?ä¸ºPoint
    ODrive.PathMove(Path1, 400, 100, 800000, 10, 1, 0);
*/

/*
    Point start_pt = {gps_x, gps_y};
    std::vector<Point> Path1 = rrtPlanner_short.optimal_rrt_planning(start_pt, (Point){48, 0, 0}, 4);  // è¿™é‡Œä¸€å®šè?å¼ºåˆ¶ç±»å‹è½¬æ?ä¸ºPoint
    ODrive.PathMove(Path1, 400, 100, 800000, 10, 1, 0);
*/

    

    // åŠ åˆ†åŒºåæ ?
    std::vector<Point> bonusAreas = {{60, 60}, {-60, 60}, {60, -60}, {-60, -60}};
    // å›ºå®šæ¡©åæ ?
    std::vector<Point> fixedStakes = {{60, 0}, {-60, 0}, {0, -60}, {0, 60}};

    // æ¸…å‰1/4åœ? 
    // å–æ¡©
    std::vector<Point> Path1 = {{-48, 24}, {-36, 36}, {-30, 40}};
    ODrive.turnToTarget((Point){-24, 48}, 40, 900, 1, 1);
    ODrive.PathMove(Path1, 400, 60, 800000, 10, 1, 0);
    ODrive.turnToTarget((Point){-24, 48}, 40, 700, 1, 1);
    ODrive.moveToTarget((Point){-24, 48}, 50, 1100, 10);
    ODrive.simpleMove(40, 180, 0.2, 10);
    task::sleep(200);
    gas_hold.state(100,pct);
    task::sleep(250);
  //  ODrive.HSAct(1, (Point){-24, 48}, 80, 80, 8000, 10, 1, 0););

     // å–äºŒç?
    ODrive.turnToTarget((Point){0, 48}, 50, 1100);
    ODrive.moveToTarget((Point){-12,52}, 40, 1000, 10);
    ODrive.HSAct(0, (Point){0, 48}, 80, 60, 1000, 15, 1, 0);
    task::sleep(500);
    ODrive.moveToTarget((Point){-25,48}, 80, 1000, 10);
    ODrive.turnToTarget((Point){0, 62}, 40, 1500);
    ODrive.HSAct(0, (Point){0, 62}, 80, 80, 8000, 15, 1, 0);
    task::sleep(500);
    ODrive.moveToTarget((Point){-39, 36}, 100, 1100, 10, 1);
    ODrive.turnToTarget((Point){-48, 48}, 40, 1200);
    ODrive.HSAct(0, (Point){-48, 48}, 1500, 80, 1500, 15, 1, 0);
     
    manual = true;
    ring_convey_spin = true;
    reinforce_stop = false;
     
    // // ===å¸å·¦ä¸‹è?’è½===
    // // å¸è?’è½çš?
    //  {
    //  ODrive.moveToTarget((Point){-53, 51}, 40, 1700, 10);
    //  //å†æ?£è½¬
    //  ring_convey_spin = true;
    //  reinforce_stop = false;
    //  ODrive.turnToAngle(-45, 50, 900);
    //  ODrive.simpleMove(30, 0, 0.4, 10);
    //  ODrive.turnToAngle(-45, 50, 900);
    //  ODrive.simpleMove(30, 0, 0.4, 10);
    //  task::sleep(150);
    //  ODrive.turnToAngle(-20, 60, 350);
    //  ODrive.simpleMove(30, 0, 0.4, 10);
    //  ODrive.turnToAngle(10, 60, 350);
    //  ODrive.simpleMove(40, 0, 0.3, 10);
    //  ODrive.simpleMove(40, 180, 0.4, 10);
    //  ODrive.simpleMove(50, 0, 0.6, 10);
    //  task::sleep(200);

    //  ODrive.turnToAngle(-45, 100, 350);

    //  }

    ODrive.moveToTarget((Point){-49, 49}, 40, 1200, 10);
    ODrive.turnToAngle(135, 75, 1500);
    ODrive.simpleMove(90, 180, 0.8, 10);
    gas_hold.state(0, pct);
  // ODrive.simpleMove(70, 0, 0.65, 10);
   // ODrive.simpleMove(100, 180, 0.75, 10);
    task::sleep(400);
     
    ODrive.HSAct(2, (Point)bonusAreas[1], 100, 80, 1500, 10, 1, 0);
    
    //ODrive.HSAct(4, (Point){-48, 48}, 100, 80, 500, 10, 1, 0);
    manual = false;
    ring_convey_spin = true;
    reinforce_stop = true;

    // æ¸…å1/4åœ? 
    // è·‘åœº
    ODrive.turnToAngle(90, 50, 1000);
    ODrive.moveToTarget((Point){-20, 48}, 60, 1200, 10);
    ODrive.moveToTarget((Point){5, 48}, 60, 1200, 10);
    ring_convey_spin = false;
    reinforce_stop = true;
    manual = false;

    // å¸åŠç?
    ODrive.turnToTarget((Point){24, 48}, 50, 1300, 1);
    ring_convey_spin = true;
    reinforce_stop = false;
    manual = false;
    ODrive.moveToTarget((Point){24, 48}, 45, 1300, 1, 1);
    ODrive.simpleMove(40, 0, 0.18, 10);
    task::sleep(500);
    ring_convey_spin = false;
    reinforce_stop = true;

    // æŒæ¡©
    ODrive.moveToTarget((Point){24, 38}, 45, 1300, 1, 1);
    ODrive.turnToTarget((Point){24, 24}, 50, 1300, 1, 1);
    ODrive.moveToTarget((Point){24, 24}, 40, 1500, 10);
    ODrive.simpleMove(40, 180, 0.12, 10);
    task::sleep(200);
    gas_hold.state(100,pct);
    task::sleep(150);
    manual = false;
    ring_convey_spin = true;
    reinforce_stop = false;
    task::sleep(550);

    // // å†²æ•£ç?(ä½¿ç”¨RRTè‡?åŠ¨è?„åˆ’è·?å¾?)
    // ODrive.moveToTarget((Point){36, 36}, 60, 800, 10);
    // ODrive.moveToTarget((Point){20, 56}, 60, 800, 10);
    // ODrive.moveToTarget((Point){59, 48}, 60, 800, 10);
    // ODrive.moveToTarget((Point){60, 40}, 60, 800, 10);

    // å¸å1/4åŒºäºŒç?
    ODrive.HSAct(0, (Point){48,24}, 60, 85, 1200, 10, 1, 0);

    // å¸åŠç?
    ODrive.moveToTarget((Point){20, 48}, 52, 500, 10);
    ODrive.moveToTarget((Point){30, 59}, 52, 1000, 10);
    ODrive.turnToTarget((Point){48, 48}, 50, 1300, 1);
    ring_convey_spin = true;
    reinforce_stop = false;
    manual = false;
    ODrive.moveToTarget((Point){48, 48}, 45, 1300, 1, 1);
    ODrive.simpleMove(45, 0, 0.3, 10);
    task::sleep(700);
    ring_convey_spin = false;
    reinforce_stop = true;
   // ODrive.HSAct(0, (Point){48, 48}, 60, 85, 700, 5, 1, 0);

    // æ¨æ¡©
    ODrive.moveToTarget((Point){50, 50}, 40, 1200, 10);
    ODrive.turnToAngle(-135, 60, 1500);
    ODrive.simpleMove(90, 180, 0.8, 10);
    gas_hold.state(0, pct);
  //  ODrive.simpleMove(70, 0, 0.65, 10);
  //  ODrive.simpleMove(100, 180, 0.75, 10);
    task::sleep(400);

    // æŒç§»åŠ¨æ¡©
    ODrive.moveToTarget((Point){40, 30}, 60, 1000, 10 , 1);
    ODrive.turnToTarget((Point){48, 0}, 40, 1000, 1, 1);
    ODrive.moveToTarget((Point){48, 0}, 40, 1200, 10 , 1);
    ODrive.simpleMove(40, 180, 0.12, 10);
    gas_hold.state(100, pct);
    task::sleep(200);
    ring_convey_spin = true;
    reinforce_stop = false;
    manual = false;
    gas_hold.state(0, pct);
    task::sleep(400);

    ODrive.moveToTarget((Point){24, 24}, 60, 1200, 10 , 1);
    ODrive.moveToTarget((Point){-4, -4}, 60, 1200, 10 , 1);


    
    while(1)
    {
            {
                convey_belt.spin(fwd,100,pct);
                roller_group.spin(fwd,-100,pct);
            }
            // »ñÈ¡µ½×î½üµÄºì»·
            int min_index;
            int min_distance = INT_MAX;
            for(int i = 0;i<my_map.size();i++){
                if(min_distance > sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y)) && my_map[i].kind == 1){
                    min_distance = sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y));
                    min_index = i;
                }
            }
            oj_data nearest_elem = my_map[min_index];
            // Í£???Ğı??
            ODrive.VRUN(0, 0, 0, 0);

            //³Ô»·
            ODrive.turnToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000);
            ODrive.moveToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000);
            task::sleep(800);
            //°Ñ»·´Ómy_map??¶ª³öµØÍ¼
            nearest_elem.x=1e5;
            nearest_elem.y=1e5;
            task::sleep(800);
            ODrive.simpleMove(80,0,1,10);
    }

//     // åŠ åˆ†åŒºåæ ?
//     std::vector<Point> bonusAreas = {{60, 60}, {-60, 60}, {60, -60}, {-60, -60}};
//     // å›ºå®šæ¡©åæ ?
//     std::vector<Point> fixedStakes = {{60, 0}, {-60, 0}, {0, -60}, {0, 60}};

//     // æ¸…å‰1/4åœ? 
//     // å–æ¡©
//     std::vector<Point> Path1 = {{-48, 12}, {-34, 29}, {-28, 42}};
//     ODrive.turnToTarget((Point){-24, 48}, 40, 900, 1, 1);
//     ODrive.PathMove(Path1, 200, 100, 800000, 10, 1, 0);
//     ODrive.turnToTarget((Point){-24, 48}, 40, 700, 1, 1);
//     ODrive.moveToTarget((Point){-24, 48}, 70, 1100, 10);
//     gas_hold.state(100,pct);

//   //  ODrive.HSAct(1, (Point){-24, 48}, 80, 80, 8000, 10, 1, 0););
//     // å–ç¯
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
     

//     // ===å¸å·¦ä¸‹è?’è½===
//     // å¸è?’è½çš?
//      {
//      ODrive.moveToTarget((Point){-51, 50}, 40, 1800, 10);
//      ODrive.turnToAngle(-43, 50, 900);
//      ODrive.simpleMove(43, 0, 0.3, 10);

//      ODrive.turnToAngle(-45, 40, 0);
    
//     //å†æ?£è½¬
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


//      // æ¸…å1/4åœ? 
//     // è·‘åœº
//     ODrive.turnToAngle(90, 75, 1000);
//     ODrive.moveToTarget((Point){-28,48}, 70, 1400, 10);
//     ODrive.moveToTarget((Point){12,48}, 70, 1400, 10);

//     // åƒç??ä¸€ä¸?åŠç¯
//     ODrive.HSAct(4, (Point){24, 48}, 75, 705, 1500, 10, 1, 0);
//     ODrive.turnToTarget((Point){24, 24}, 60, 1100, 1, 1);

//     // æŒæ¡©
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



    // åŠ¨ä½œç©ºé—´:0å–ç¯, 1å–æ¡©, 2æ”¾æ¡©, 3æ‰£ç¯, 4å–åŠç? 





    // å¥—è¾¹æ¡? 
   // ODrive.HSAct(3, (Point)fixedStakes[2], 75, 100, 1500, 10, 1, 0);
    //imu.setHeading(GPS_.heading(deg), deg);

   
    //thread go_out_roller(go_out_roller);
    //ODrive.HSAct(4, (Point){7, -7}, 60, 80, 1000, 10, 1, 0);
    //task::sleep(500);



}
/***************************
 
      skillautonomous run

 **************************/
void skillautonoumous(){
   
}
/***************************
 
      usercontrol run

 **************************/
void usercontrol()
{
    Controller1.ButtonL1.pressed([]() {
        lift_arm.spin(forward); // ç”µæœºæ­£è½¬
    });

    Controller1.ButtonL1.released([]() {
        lift_arm.setStopping(brakeType::hold);
        lift_arm.stop(hold);
    });

    Controller1.ButtonL2.pressed([]() {
         lift_arm.spin(vex::reverse); // ç”µæœºåè½¬
    });

    Controller1.ButtonL2.released([]() {
        lift_arm.setStopping(brakeType::hold);
        lift_arm.stop(hold);
    });
    Controller1.ButtonR1.pressed([]() {
        static bool motorRunning = false; // ç”¨äºè¿½è¸ªç”µæœºçŠ¶æ€?
        
        if (!motorRunning) {
             manual = true;
             ring_convey_spin = true;
             reinforce_stop = false;
            //roller_group.spin(reverse, 100, pct);
            //convey_belt.spin(forward, 100, pct);

        } else {
            manual = true;
            reinforce_stop = true;
           roller_group.stop();// åœæ?¢ç”µæœºæ—‹è½?
           convey_belt.stop();
        }
        motorRunning = !motorRunning; // åˆ‡æ¢ç”µæœºçŠ¶æ€}
    });

    Controller1.ButtonR2.pressed([]() {
        static bool motorRunning = false; // ç”¨äºè¿½è¸ªç”µæœºçŠ¶æ€?

        if (!motorRunning) {
            manual = true;
            reinforce_stop = false;
            roller_group.spin(reverse,100,pct);
            convey_belt.spin(forward,100,pct);
            
        } else {
            manual = true;
            reinforce_stop = false;

            roller_group.stop();// åœæ?¢ç”µæœºæ—‹è½?
            convey_belt.stop();
            

        }
        motorRunning = !motorRunning; // åˆ‡æ¢ç”µæœºçŠ¶æ€}
    });


    Controller1.ButtonL1.released([]() {
        lift_arm.setStopping(brakeType::hold);
        lift_arm.stop(hold);
    });


    Controller1.ButtonB.pressed([]() {
         static bool status_hold = false; // ç”¨äºè¿½è¸ªç”µæœºçŠ¶æ€?

         if (!status_hold) {
             gas_hold.state(100,pct);
         } else {
             gas_hold.state(0,pct);
         }
         status_hold = !status_hold; // åˆ‡æ¢çŠ¶æ€?
     });



    while(true){
        
        ODrive.ManualDrive_nonPID();

        // è°ƒè¯•æ—¶é€šè¿‡æŒ‰é”®è¿›å…¥è‡?åŠ?
         if(Controller1.ButtonX.pressing()){ 
             autonomous();
         }
         if(Controller1.ButtonY.pressing()){
             skillautonoumous();
         }

        if(Controller1.ButtonUp.pressing()){
            vexMotorVoltageSet(side_bar.index(), 100*120);
        }else if(Controller1.ButtonDown.pressing()){
            vexMotorVoltageSet(side_bar.index(), -100*120);
        }else{
            side_bar.stop(hold);
        }

    }
}



int main() {

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
  this_thread::sleep_for(loop_time);



//  Competition.drivercontrol(usercontrol);

  // Run the pre-autonomous function.
  pre_auton();

    Competition.autonomous(autonomous);

  // Prevent main from exiting with an infinite loop.
  bool gps_jetson_nano_dead = false;
  while(1) {
    my_map = {};
    gps_jetson_nano_dead = false;
      // get last map data
      jetson_comms.get_data( &local_map );

        // ĞèÒª???jetson nano´¦µÄGPSËÀµô×öÒ»??Ó¦¼±??????
        if(fabs(local_map.pos.x - 0) < 1e-6 && fabs(local_map.pos.y - 0) < 1e-6 && fabs(local_map.pos.rot - 0) < 1e-6){       // ÍêÈ«??(0, 0)??ÓĞÔÚËÀµôµÄÇé¿öÏÂ²Å¿ÉÄÜ³ö??
            gps_jetson_nano_dead = true;

        }
        // ¸ĞÖªµ½µÄÒÆ¶¯³¡µØÔªËØµÄ±¾µØÄÚ´æ´æ??
        for(int i=0;i<local_map.detectionCount;i++)
        {

            oj_data data;

            T data_x, data_y;
            if(!gps_jetson_nano_dead){

                data_x = local_map.detections[i].mapLocation.x * 39.3700788;
                data_y = -2+local_map.detections[i].mapLocation.y * 39.3700788;
            }else{  // jetson_nano¶ÁGPS³öÏÖÁËÎÊÌâ£¬Ò»Ö±ÊÇ(0, 0), ÔòĞèÒªÀûÓÃ±¾µØĞÅ??????
                T local_data_x = local_map.detections[i].mapLocation.x * 39.3700788;
                T local_data_y = local_map.detections[i].mapLocation.y * 39.3700788;
                T sum_offset_x =  local_data_x;   
                T sum_offset_y = local_data_y;

                T theta = GPS_.heading(deg) / 180 * 3.1415926535;
                data_x = gps_x + ( sum_offset_y * sin(theta) + sum_offset_x * cos(theta) );
                data_y = gps_y + ( sum_offset_y * cos(theta) - sum_offset_x * sin(theta) );
            }

            //x y ×ø±ê : Ïà???ÓÚ??¼ºÎ»??+??¼ºµÄÎ»ÖÃ
            data.x = data_x;
            data.y = data_y;
            // Àà±ğ
            data.kind = local_map.detections[i].classID; 
            my_map.push_back(data);

        }
                // µ÷ÊÔÊ±Í¨¹ı°´¼ü½øÈë????
         if(Controller1.ButtonX.pressing()){ 
             autonomous();
         }

#ifdef JETSON_NANO_VISION_DEBUG
        // jetson_nano¸ĞÖªµ÷ÊÔ
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

