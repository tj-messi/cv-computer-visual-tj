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
#include <vector>

using namespace vex;
using namespace tjulib;

/*---------------  模式选择  ---------------*/
// 如果进行技能赛就def，否则注释，进行自动
//#define SKILL
// 如果用里程计就def，否则注释，用雷达
#define ODOM
// 如果要开启远程调试就def，否则就注释
#define Remotedeubug


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
// Dual-Communication Demo
void demo_dualCommunication(){
    sendTask();  // 向联队车发送信息
    task::sleep(200);
    Brain.Screen.print("send thread jump out\n");

    /************************
      
      发送完信号后执行的程序
      
    ************************/

    // 等待一下
    while(1){
        AllianceLink.received("finished", confirm_SmallCar_Finished);
        task::sleep(200);
        if(received_flag){break;}
    }

}


void auto_Isolation(void) {
    //while(1)
    {
        if(local_map.detectionCount>0)
        {
            convey_belt.spin(fwd);
        }
    }
}
void auto_Interaction(void) {
    while(1)
    {
        //if(local_map.detectionCount>0)
        {
            convey_beltMotorA.setVelocity(60,percent);
            convey_beltMotorB.setVelocity(60, percent);
        }
    }
}
bool firstAutoFlag = true;
void autonomousMain(void) {


  if(firstAutoFlag)
    auto_Isolation();
  else 
    auto_Interaction();

  firstAutoFlag = false;
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
void create_map(double diff = 0.05,AI_RECORD local_map)
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

  R.push_back({0,local_map.pos.x,local_map.pos.y});//起始点

  int hold_count = 0;

  while(1) {
      // get last map data
      jetson_comms.get_data( &local_map );

      // set our location to be sent to partner robot
      link.set_remote_location( local_map.pos.x, local_map.pos.y, local_map.pos.az, local_map.pos.status );

      // fprintf(fp, "%.2f %.2f %.2f\n", local_map.pos.x, local_map.pos.y, local_map.pos.az)

      // request new data    
      // NOTE: This request should only happen in a single task.    
      jetson_comms.request_map();


      // ai-decision  
      {
        create_map(0.3,local_map);
        init_dp();
        run();
        tt=0;
        path.clear();
        getPath(0);
        int target_ring=paths[0][paths[0].size()-1];
        double timecost=sqrt(pow(R[target_ring].x-local_map.pos.x,2)+pow(R[target_ring].y-local_map.pos.y,2));
        ODrive.moveToTarget({R[target_ring].x,R[target_ring].y,0});
        R[target_ring].x=1e5+5;
        R[target_ring].y=1e5+5;
        // 等待到达目标点
        this_thread::sleep_for(timecost*1000);
      }  


      {
        if(hold_count==6)
        {
            ODrive.moveToTarget({endx,endy,0});
            hold_count=0;
        }
      }
      // Allow other tasks to run
      this_thread::sleep_for(loop_time);
  }
}