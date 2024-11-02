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
// 如果进行技能赛就def，否则注释，进行自动
//#define SKILL
// 如果用里程计就def，否则注释，用雷达
#define ODOM
// 如果要开启远程调试就def，否则就注释
#define Remotedeubug
// 如果是红方就def，否则就注释
#define RED

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
// 红方标志
bool is_red = true;
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

pidParams   fwd_pid(6.5, 0.3, 0.3, 2, 2.5, 7, 15), 
            turn_pid(2, 0.15, 0.15, 45, 1, 5, 15), 
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
HighStakeMap map(PosTrack->position, &local_map);

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
            // 第4秒的时候会更新一下坐标
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
#ifdef RED
    is_red = true;
#else
    is_red = false;
#endif
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
    thread receive(receivedTask);

    // 这里考虑到只使用imu而不使用gps的情况
    if(imu.installed()){
        // 设置初始位置
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


// Dual-Communication Demo

int push_enemyring_thread(){
    task::sleep(2500);
    gas_push.state(100, pct);
    task::sleep(1000);
    gas_push.state(0, pct);
    return 0;
}
int push_enemyring_thread2(){
    task::sleep(5000);
    gas_push.state(100, pct);
    task::sleep(2000);
    gas_push.state(0, pct);
    return 0;
}
int push_enemyring_thread3(){
    gas_push.state(100, pct);
    task::sleep(1600);
    gas_push.state(0, pct);
    return 0;
}

int go_out_roller(){
    while(1){
        manual = true;
        reinforce_stop = true;
        // 当进入桩内的时候退出
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
        // 当进入桩内的时候退出
        if((gps_x * gps_x + gps_y * gps_y) >= 28){
            manual = true;
            ring_convey_spin = true;
            reinforce_stop = false;
            break;
        }
    }
    return 0;
}

const int MAX_NUM_RINGS = 15;
	double a[MAX_NUM_RINGS][MAX_NUM_RINGS];
	double dp[MAX_NUM_RINGS][1<<MAX_NUM_RINGS];
	int t;
	//自建图
	struct ring
	{
	    int col;//0--red, 1--blue
	    double x,y;
	};

std::vector<ring> R;
	void create_map(AI_RECORD local_map,double diff = 0.05)
	{
	    int num = R.size();
	    for(int j=0;j<local_map.detectionCount;j++)
	    {    
	        double nx = local_map.detections[j].mapLocation.x+local_map.pos.x;
	        double ny = local_map.detections[j].mapLocation.y+local_map.pos.y;
	        int col = local_map.detections[j].classID;
	        for(int i=0;i<num;i++)
	        {
	            if(abs(nx-R[i].x)>diff || abs(ny-R[i].y)>diff)
	            {
	                R.push_back({col,nx,ny});
	                break;
	            }
	        }
	    }
	}

void init_dp()
	{
	    int n=R.size();
	    int m=n*(n-1)/2;//完全图的边数
	    memset(dp,0,sizeof(dp));
	    memset(a,0,sizeof(a));
	    for(int i=0;i<m;i++)
	    {
	        for(int j=0;j<m;j++)
	        {
	            a[i][j]=sqrt((R[i].x-R[j].x)*(R[i].x-R[j].x)+(R[i].y-R[j].y)*(R[i].y-R[j].y));
	            a[j][i]=a[i][j];
	            //双向边
	        }
	    }
	}


void run()
	{
	    //dp核心算法
	    int n=R.size();
	    int m=n*(n-1)/2;//完全图的边数 
		t=(1<<n);
		for(int i=1;i<n;i++){//因为起点初始不能被标记已经走过,所以需要手动初始化起点到达其它点的花费 
			dp[i][1<<i]=a[0][i];
		}
		for(int i=0;i<t;i++){//枚举每一个状态 
			for(int j=0;j<n;j++){//枚举每一个没有走过的点 
				if(((i>>j)&1)==0){
					for(int k=0;k<n;k++){//枚举每一个走过的点 
						if(((i>>k)&1)==1&&dp[j][i^(1<<j)]>dp[k][i]+a[k][j]){//取最优状态 
							dp[j][i^(1<<j)]=dp[k][i]+a[k][j];
						}
					}
				}
			}
		}
	}    
void autonomous(){
/*
    Point start_pt = {gps_x, gps_y};
    std::vector<Point> Path1 = rrtPlanner.optimal_rrt_planning(start_pt, (Point){48, 0, 0}, 4);  // 这里一定要强制类型转换为Point
    ODrive.PathMove(Path1, 100, 100, 800000, 10, 1, 0);
*/
    
    // 加分区坐标
    std::vector<Point> bonusAreas = {{62, 62}, {-62, 62}, {62, -62}, {-62, -62}};
    // 固定桩坐标
    std::vector<Point> fixedStakes = {{60, 0}, {-60, 0}, {0, -60}, {0, 60}};
    
    
    // 动作空间:0取环, 1取桩, 2放桩, 3扣环, 4取半环 
    // 清前1/4场 
     //ODrive.turnToTarget((Point){-48,-48}, 70, 1000); 
     ODrive.HSAct(4, (Point){-48,-48}, 60, 85, 1500, 25, 1, 0);
     ODrive.setStop(hold);
     ODrive.turnToTarget((Point){-24,-48},70,1000,1,1);
     ODrive.simpleMove(70, 180, 0.45);
     ODrive.turnToTarget((Point){-24,-48}, 70,1100,1,1);
     ODrive.moveToTarget((Point){-24,-48}, 100, 1200, 10);

     gas_hold.state(80, pct);
     task::sleep(300);

     manual = true;
     ring_convey_spin = true;
     reinforce_stop = false;
     ODrive.turnToTarget((Point){-24,-24}, 55, 1000); 
     ODrive.moveToTarget((Point){-24,-30}, 80, 1200);
     ODrive.turnToTarget((Point){-24,-20}, 55, 1200); 
     ODrive.simpleMove(70, 0, 0.25, 10);

    // 吃二环
     ODrive.turnToTarget((Point){0,-40}, 65, 900); 
     ODrive.moveToTarget((Point){-10,-40}, 80, 1000);
     ODrive.HSAct(0, (Point){0,-45}, 60, 70, 8000, 15, 1, 0);
     ODrive.moveToTarget((Point){-12,-48}, 70, 1200);
     thread push_enemyring(push_enemyring_thread);
     manual = false;
     ring_convey_spin = true;
     reinforce_stop = false;
     ODrive.HSAct(0, (Point){0,-56}, 60, 85, 8000, 25, 1, 0);
     // 吸角落的
     {
     ODrive.moveToTarget((Point){-50.5, -50}, 80, 1600, 10);
     ODrive.turnToAngle(225, 80, 1100);
     ODrive.simpleMove(100, 0, 1, 10);
     manual = true;
     ring_convey_spin = true;
     reinforce_stop = false;
     ODrive.turnToAngle(155, 85, 1100);
     ODrive.simpleMove(50, 180, 0.2, 10);
     ODrive.turnToAngle(180, 85, 1100);
     ODrive.simpleMove(70, 0, 0.4, 10);
     ODrive.simpleMove(50, 180, 0.2, 10);

        {
            //ODrive.simpleMove(70, 0, 1, 10);
            ODrive.simpleMove(90, 0, 1, 10);
        }
     }
     task::sleep(500);
    ODrive.HSAct(2, (Point)bonusAreas[3], 75, 100, 1200, 10, 1, 0);
    // 套边桩 
  
    //thread push_enemyring2(push_enemyring_thread2);

    //ODrive.HSAct(3, (Point)fixedStakes[2], 75, 100, 1500, 10, 1, 0);

    // 清后1/4场
    // 跑场
    ODrive.turnToAngle(90, 80, 900);
    ODrive.moveToTarget((Point){-25,-40}, 100, 1000, 10);
    ODrive.moveToTarget((Point){12,-40}, 100, 1000, 10);

    ODrive.setStop(hold);
    imu.setHeading(GPS_.heading(deg), deg);
    task::sleep(300);

    ODrive.HSAct(4, (Point){24, -48}, 75, 75, 1500, 10, 1, 0);
    // 持桩
    ODrive.moveToTarget((Point){10,-48}, 70, 800, 10);
    ODrive.setStop(hold);
    ODrive.HSAct(1, (Point){24, -24}, 70, 85, 1000, 15, 1, 0);    
    ring_convey_spin = true;
    reinforce_stop = false;
    task::sleep(150);
    //thread go_out_roller(go_out_roller);
    //ODrive.HSAct(4, (Point){7, -7}, 60, 80, 1000, 10, 1, 0);
    //task::sleep(500);

    ODrive.HSAct(0, (Point){48,-24}, 60, 85, 1200, 10, 1, 0);
    ODrive.moveToTarget((Point){20,-48}, 92, 500, 10);
    ODrive.moveToTarget((Point){32,-58}, 92, 1000, 10);
    ODrive.HSAct(0, (Point){48,-48}, 60, 85, 700, 5, 1, 0);
     // 吸角落的
     {
     ring_convey_spin = true;
     reinforce_stop = false;
     thread push_enemyring_thread4(push_enemyring_thread);
     ODrive.moveToTarget((Point){52
     ,-50}, 80, 1200, 10);
     ODrive.turnToAngle(135, 100, 1200); 

     thread push_enemyring3(push_enemyring_thread3);
     ring_convey_spin = true;
     reinforce_stop = false;
     ODrive.simpleMove(60, 0, 1, 10);
     task::sleep(600);
     // 转正
     ODrive.turnToAngle(150, 40, 400); 
     ODrive.simpleMove(60, 0, 0.3, 10);
     ODrive.turnToAngle(165, 40, 400); 
     ODrive.simpleMove(60, 0, 0.3, 10);
     ODrive.turnToAngle(180, 40, 500); 
     ODrive.simpleMove(70, 0, 0.3, 10);
     ODrive.turnToAngle(180, 40, 500); 
     ODrive.simpleMove(40, 0, 0.4, 10);
     ODrive.simpleMove(40, 180, 0.4, 10);
     ODrive.simpleMove(40, 0, 0.4, 10);

     ring_convey_spin = true;
     reinforce_stop = false;
     task::sleep(300);
     }
    ODrive.HSAct(2, (Point)bonusAreas[2], 75, 100, 1200, 10, 1, 0);

    // // 清移动桩
    //  ODrive.moveToTarget((Point){16,-40}, 100, 800, 10);
    //  ODrive.HSAct(1, (Point){46, 0}, 60, 85, 1000, 20, 1, 0);
    //  ODrive.HSAct(0, (Point){56, 0}, 60, 85, 1000, 20, 1, 0);
    //  ODrive.turnToAngle(225, 50, 1500); 
    //  gas_hold.state(0, pct);

    // 挂
    // Point start_pt = {gps_x, gps_y};
    // std::vector<Point> Path1 = rrtPlanner_short.rrt_planning(start_pt, (Point){7, -20, 0});  // 这里一定要强制类型转换为Point
    // ODrive.PathMove(Path1, 100, 100, 800000, 10, 1, 0);
   // ODrive.moveToTarget((Point){40,-40}, 100, 2000, 10);

    ODrive.moveToTarget((Point){50,-50}, 100, 2000, 10);
    ODrive.turnToAngle(-55, 100, 1200); 
    gas_lift.state(100, pct);
    ODrive.VRUN(100, 100, -100, -100); 
    task::sleep(2300);
    ODrive.VRUN(0, 0, 0, 0); 

    
}

int tt;//记录 
	std::vector<int> path(1,0);//初始化从0点出发 ,存储单条路径 
	std::vector<std::vector<int> > paths;//存储所有的路径 
	void getPath(int p){//递归查找所有路径 
	    int n=R.size();
	    int m=n*(n-1)/2;//完全图的边数 
		if((tt^(1<<p))==0){//如果是最后一个点了就存储改路径 
			paths.push_back(path);
			return; 
		}
		for(int j=1;j<n;j++){
			//回溯算法，一个加法的原则
			//如果点1到达点5的最短距离为100，点1到达点3的最短距离是70
			//而点3和点5之间的距离为30 ，那么点3是点1到5之间的一个中间点
			//即1-->...-->3-->5 
			if(a[j][p]+dp[j][tt^(1<<p)]==dp[p][tt]){
				tt^=(1<<p);
				path.push_back(j);
				getPath(j);
				tt^=(1<<p);
				path.pop_back();
			}
		}
		
	}
struct oj_data
{
    double x;
    double y;
    int kind;
};
static std::vector<oj_data> my_map;
void autonomons(){

    /*===================================================================================================
    
        定义了一套Action函数
        动作空间 
        取环  ： 0  ODrive.HSAct(0, ring_point, gola_pt_track_time, maxSpeed, maxtime_ms, gaptime, fwd);
        取桩  ： 1  ODrive.HSAct(1, stake_point, gola_pt_track_time, maxSpeed, maxtime_ms, gaptime, fwd);
        放桩  ： 2  ODrive.HSAct(2, stake_point, gola_pt_track_time, maxSpeed, maxtime_ms, gaptime, fwd);
        扣环  ： 3  ODrive.HSAct(3, fixed_stake_point, gola_pt_track_time, maxSpeed, maxtime_ms, gaptime, fwd);
        取半环 ：4  ODrive.HSAct(4, ring_point, gola_pt_track_time, maxSpeed, maxtime_ms, gaptime, fwd);
        
    
    =======================================================================================================*/
    
    /* ------- AI 感知清理前1/4场 --------- */
    
    int area = gps_x/36+gps_y/36;
    int edge_x = 0;
    int edge_y = 0;
    switch(area)
    {
        case 0:
            // 左上右下
            if(gps_x/36<0)
            {
                //左上
                edge_x = -36;
                edge_y = 36;
            }
            else
            {
                //右下
                edge_x = 36;
                edge_y = -36;
            }
            
            break;
        case 2:
            //右上
            edge_x = 36;
            edge_y = 36;
            
            break;
        case -2:
            //左下
            edge_x = -36;
            edge_y = -36;
            break;
    }

    while(1)
        {   
            // {
            //     convey_belt.stop();
            //     roller_group.stop();
            // }
            // 鑾峰彇鍒版渶杩戠殑鏌卞瓙
            int min_index;
            int min_distance = INT_MAX;
            for(int i = 0;i<my_map.size();i++){
                if(min_distance > sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y)) && my_map[i].kind == 0){
                    min_distance = sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y));
                    min_index = i;
                }
            }
            oj_data nearest_elem = my_map[min_index];
            // 鍋滄?㈡棆杞?
            ODrive.VRUN(0, 0, 0, 0);
            // 鏈濆悜鐜?
            ODrive.turnToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000,1,1);
            // 鍚冪幆
            ODrive.moveToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000);
            ODrive.simpleMove(80,180,1,10);
            gas_hold.state(100,pct);
           //寰€鍓嶈蛋涓€娈佃窛绂?
           task::sleep(800);
           ODrive.simpleMove(80,0,1,10);

           break;
        }


    {
        create_map(local_map,0.3);
        init_dp();
        run();
        tt=0;
        path.clear();
        getPath(0);
        int target_ring=paths[0][paths[0].size()-1];
        double timecost=sqrt(pow(R[target_ring].x-local_map.pos.x,2)+pow(R[target_ring].y-local_map.pos.y,2));
        ODrive.HSAct(0, {R[target_ring].x,R[target_ring].y,0});
        R[target_ring].x=1e5+5;
        R[target_ring].y=1e5+5;
        // 等待到达目标点
        this_thread::sleep_for(timecost*1000);
    }
    /* ---------- 吸角落的环 --------------*/
{
     {
     ODrive.moveToTarget((Point){-50.5, -50}, 80, 1600, 10);
     ODrive.turnToAngle(225, 80, 1100);
     ODrive.simpleMove(100, 0, 1, 10);
     manual = true;
     ring_convey_spin = true;
     reinforce_stop = false;
     ODrive.turnToAngle(155, 85, 1100);
     ODrive.simpleMove(50, 180, 0.2, 10);
     ODrive.turnToAngle(180, 85, 1100);
     ODrive.simpleMove(70, 0, 0.4, 10);
     ODrive.simpleMove(50, 180, 0.2, 10);
     ODrive.simpleMove(70, 0, 1, 10);
     }
     task::sleep(500);
     // 放桩
     ODrive.HSAct(2, (Point){map.bonusAreas[3].x, map.bonusAreas[3].y}, 75, 100, 1200, 10, 1, 0);
}
    /* ---------- 跑场 --------------*/
{
    ODrive.turnToAngle(90, 80, 900);
    ODrive.moveToTarget((Point){-25,-40}, 100, 1000, 10);
    ODrive.moveToTarget((Point){12,-40}, 100, 1000, 10);
}
    /* ------- AI 感知清理后1/4场 --------- */
     while(1)
        {   
            // {
            //     convey_belt.stop();
            //     roller_group.stop();
            // }
            // 鑾峰彇鍒版渶杩戠殑鏌卞瓙
            int min_index;
            int min_distance = INT_MAX;
            for(int i = 0;i<my_map.size();i++){
                if(min_distance > sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y)) && my_map[i].kind == 0){
                    min_distance = sqrt((gps_x - my_map[i].x) * (gps_x - my_map[i].x) + (gps_y - my_map[i].y));
                    min_index = i;
                }
            }
            oj_data nearest_elem = my_map[min_index];
            // 鍋滄?㈡棆杞?
            ODrive.VRUN(0, 0, 0, 0);
            // 鏈濆悜鐜?
            ODrive.turnToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000,1,1);
            // 鍚冪幆
            ODrive.moveToTarget(Point{nearest_elem.x, nearest_elem.y}, 80, 2000);
            ODrive.simpleMove(80,180,1,10);
            gas_hold.state(100,pct);
           //寰€鍓嶈蛋涓€娈佃窛绂?
           task::sleep(800);
           ODrive.simpleMove(80,0,1,10);

           break;
        }


    {
        create_map(local_map,0.3);
        init_dp();
        run();
        tt=0;
        path.clear();
        getPath(0);
        int target_ring=paths[0][paths[0].size()-1];
        double timecost=sqrt(pow(R[target_ring].x-local_map.pos.x,2)+pow(R[target_ring].y-local_map.pos.y,2));
        ODrive.HSAct(0, {R[target_ring].x,R[target_ring].y,0});
        R[target_ring].x=1e5+5;
        R[target_ring].y=1e5+5;
        // 等待到达目标点
        this_thread::sleep_for(timecost*1000);
    }
    /* ---------- 吸角落的环 --------------*/      
{
    // 吸角落的
     {
     ring_convey_spin = true;
     reinforce_stop = false;
     thread push_enemyring_thread4(push_enemyring_thread);
     ODrive.moveToTarget((Point){52
     ,-50}, 80, 1200, 10);
     ODrive.turnToAngle(135, 100, 1200); 

     thread push_enemyring3(push_enemyring_thread3);
     ring_convey_spin = true;
     reinforce_stop = false;
     ODrive.simpleMove(60, 0, 1, 10);
     task::sleep(600);
     // 转正
     ODrive.turnToAngle(150, 40, 400); 
     ODrive.simpleMove(60, 0, 0.3, 10);
     ODrive.turnToAngle(165, 40, 400); 
     ODrive.simpleMove(60, 0, 0.3, 10);
     ODrive.turnToAngle(180, 40, 500); 
     ODrive.simpleMove(70, 0, 0.3, 10);
     ODrive.turnToAngle(180, 40, 500); 
     ODrive.simpleMove(40, 0, 0.4, 10);
     ODrive.simpleMove(40, 180, 0.4, 10);
     ODrive.simpleMove(40, 0, 0.4, 10);

     ring_convey_spin = true;
     reinforce_stop = false;
     task::sleep(300);
     }
    ODrive.HSAct(2, (Point){map.bonusAreas[2].x, map.bonusAreas[2].y}, 75, 100, 1200, 10, 1, 0);
}

}

void skillautonoumous(){
   
}

void usercontrol()
{
    Controller1.ButtonL1.pressed([]() {
        lift_arm.spin(forward); // 电机正转
    });

    Controller1.ButtonL1.released([]() {
        lift_arm.setStopping(brakeType::hold);
        lift_arm.stop(hold);
    });

    Controller1.ButtonL2.pressed([]() {
         lift_arm.spin(vex::reverse); // 电机反转
    });

    Controller1.ButtonL2.released([]() {
        lift_arm.setStopping(brakeType::hold);
        lift_arm.stop(hold);
    });
    Controller1.ButtonR1.pressed([]() {
        static bool motorRunning = false; // 用于追踪电机状态
        
        if (!motorRunning) {
            manual = true;
            ring_convey_spin = true;
            reinforce_stop = false;
           // //roller_group.spin(forward, 100, pct);
               // convey_belt.spin(forward, 100, pct);

        } else {
            manual = true;
            reinforce_stop = true;
           roller_group.stop();// 停止电机旋转
           convey_belt.stop();
        }
        motorRunning = !motorRunning; // 切换电机状态}
    });

    Controller1.ButtonR2.pressed([]() {
        static bool motorRunning = false; // 用于追踪电机状态

        if (!motorRunning) {
            manual = true;
            reinforce_stop = false;
            roller_group.spin(forward,-100,pct);
            convey_belt.spin(reverse,100,pct);
            
        } else {
            manual = true;
            reinforce_stop = false;

            roller_group.stop();// 停止电机旋转
            convey_belt.stop();
            

        }
        motorRunning = !motorRunning; // 切换电机状态}
    });


    Controller1.ButtonL1.released([]() {
        lift_arm.setStopping(brakeType::hold);
        lift_arm.stop(hold);
    });

     Controller1.ButtonY.pressed([]() {
         static bool status_push = false; // 用于追踪电机状态

         if (!status_push) {
             gas_push.state(100,pct);
         } else {
             gas_push.state(0,pct);
         }
         status_push = !status_push; // 切换状态
     });

    Controller1.ButtonB.pressed([]() {
         static bool status_hold = false; // 用于追踪电机状态

         if (!status_hold) {
             gas_hold.state(100,pct);
         } else {
             gas_hold.state(0,pct);
         }
         status_hold = !status_hold; // 切换状态
     });

     Controller1.ButtonDown.pressed([]() {
         static bool status_lift = false; // 用于追踪电机状态

         if (!status_lift) {
             gas_lift.state(100,pct);
         } else {
             gas_lift.state(0,pct);
         }
         status_lift = !status_lift; // 切换状态
     });

    while(true){
        
        ODrive.ManualDrive_nonPID();

        // 调试时通过按键进入自动
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
  FILE *fp = fopen("/dev/serial2","wb");
  this_thread::sleep_for(loop_time);

  #ifdef SKILL
  Competition.autonomous(skillautonoumous);
  #else

    Competition.autonomous(autonomous);

  #endif


//  Competition.drivercontrol(usercontrol);

  // Run the pre-autonomous function.
  pre_auton();

  // Prevent main from exiting with an infinite loop.
  while (true) {
        // get last map data
      jetson_comms.get_data( &local_map );

      // set our location to be sent to partner robot
      link.set_remote_location( local_map.pos.x, local_map.pos.y, local_map.pos.az, local_map.pos.status );

     // printf("%.2f %.2f %.2f\n", local_map.pos.x, local_map.pos.y, local_map.pos.az);
     {
        map.UpdatePos();
     }
      // request new data    
      // NOTE: This request should only happen in a single task.    
      jetson_comms.request_map();

      // Allow other tasks to run
      this_thread::sleep_for(loop_time);
  }
}

