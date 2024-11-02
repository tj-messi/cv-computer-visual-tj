#pragma once
#include "vex.h"
#include "tjulib-uppercontrol/ConveyRing.hpp"
#include "tjulib-uppercontrol/LiftArm.hpp"
#include "tjulib-chassis/oct-chassis/oct-cur.hpp"
#include "tjulib-motionplanner/tjulib-pathplanner/rrt.hpp"
#include "tjulib-map/HighStakeMap.hpp"
#include <string>
#include <iostream>
using namespace vex;
typedef double T;

namespace tjulib
{
    class Oct_Action : virtual public Oct_CurChassis{
    private:
        const double PI = 3.14159265358979323846;
        
    public:
        
        HighStakeMap *map = NULL;
        RRT *rrt_short = NULL;
        RRT *rrt_long = NULL;
        Oct_Action(std::vector<std::vector<vex::motor*>*>& _chassisMotors, pidControl* _motorpidControl, Position* _position, const T _r_motor, pidControl* _fwdpid, pidControl* _turnpid, PurePursuit *_ppcontrol, HighStakeMap *_map, RRT *_rrt_short, RRT *_rrt_long) :
            Oct_BaseChassis(_chassisMotors, _motorpidControl, _position, _r_motor), 
            Oct_StraChassis(_chassisMotors, _motorpidControl, _position, _r_motor, _fwdpid, _turnpid),
            Oct_CurChassis(_chassisMotors, _motorpidControl, _position, _r_motor, _fwdpid, _turnpid, _ppcontrol),
            map(_map), rrt_short(_rrt_short), rrt_long(_rrt_long) {}
        

        void 
        HSAct(int action_index, Point target, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 15000, T move_gaptime = 10 ,int fwd = 1, int back = 0){
            switch(action_index){
             case 0:
                back = 0;   // 正着取环
                MoveForRing(target, goal_pt_track_time, maxSpeed, maxtime_ms, move_gaptime ,fwd, back);
                break;
            case 1:
                back = 1;   // 倒着取桩
                MoveForStake(target, goal_pt_track_time, maxSpeed, maxtime_ms, move_gaptime ,fwd, back);
                break;
            case 2:         // 倒着放桩
                back = 1;
                LayDownStake(target, goal_pt_track_time, maxSpeed, maxtime_ms, move_gaptime ,fwd, back);
                break;
            case 3:         // 倒着扣环
                back = 1;
                SlamDownRing(target, goal_pt_track_time, maxSpeed, maxtime_ms, move_gaptime ,fwd, back);
                break;
            case 4:         // 正着取环(取一半)
                back = 0;
                MoveForRingHalf(target, goal_pt_track_time, maxSpeed, maxtime_ms, move_gaptime ,fwd, back);
                break;
            default:
                break;
            }
           
        }


        // 取环
        void MoveForRing(Point target, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 15000, T move_gaptime = 10 ,int fwd = 1, int back = 0){
            RRT _rrt_long = *rrt_long; 
            RRT _rrt_short = *rrt_short;
            
            /*= Step1 : 如果机器人在操作半径外则需要先到达可执行的操作半径范围内 =*/
            const T radius = 8; 

            if(fabs(GetDistance(position->globalPoint, target)) >= radius * 10){  
                std::vector<Point> path;
                if(GetDistance(position->globalPoint, target) > 55){
                    /*= Step1.1 : 计算轨迹与圆的交点 =*/
                     Point target_pt = CalcuTargetPoint(_rrt_long, target, radius);
                     target_pt.angle = 0;
                    /*= Step1.2 : RRT规划器获取轨迹 =*/
                    printf("rrt2_start\n");   
                    path = _rrt_long.optimal_rrt_planning({position->globalPoint.x, position->globalPoint.y, 0}, (Point)(target_pt), 4);  
                     for(auto point : path){
                         printf("{x:%lf , y:%lf}\n", point.x, point.y);
                     }
                    printf("x0 : %lf, y0 : %lf \n", gps_x, gps_y);
                    printf("x : %lf, y : %lf \n", target_pt.x, target_pt.y);
                    printf("rrt2_end\n");
                }else{
                    /*= Step1.1 : 计算轨迹与圆的交点 =*/
                    Point target_pt = CalcuTargetPoint(_rrt_short, target, radius);
                     target_pt.angle = 0;
                    /*= Step1.2 : RRT规划器获取轨迹 =*/
                    printf("rrt2_start\n");   
                    path = _rrt_short.optimal_rrt_planning({position->globalPoint.x, position->globalPoint.y, 0}, (Point)(target_pt), 4);  
                     for(auto point : path){
                         printf("{x:%lf , y:%lf}\n", point.x, point.y);
                     }
                    printf("x0 : %lf, y0 : %lf \n", gps_x, gps_y);
                    printf("x : %lf, y : %lf \n", target_pt.x, target_pt.y);
                    printf("rrt2_end\n");
                }
               
                /*= Step1.3 : 规划路径跟踪 =*/
                PathMove(path, goal_pt_track_time, maxSpeed, maxtime_ms, move_gaptime, fwd, back);
            }

            //printf("11111111111111\n"); 
            /*= Step2 :执行最后一步动作 =*/
            // 开启吸环线程
            manual = false;
            ring_convey_spin = true;
            reinforce_stop = false;
            turnToTarget(target,  maxSpeed * 0.7, 1100,  fwd, back);

            moveToTarget(target, 80, 1200, 15);            
            simpleMove(90, 0, 0.28, 10);
            // 时间用于套环
            this_thread::sleep_for(300);

        }

        // 取半环
        void MoveForRingHalf(Point target, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 15000, T move_gaptime = 10 ,int fwd = 1, int back = 0){
            RRT _rrt_long = *rrt_long; 
            RRT _rrt_short = *rrt_short;  
            /*= Step1 : 如果机器人在操作半径外则需要先到达可执行的操作半径范围内 =*/
            const T radius = 8; 

            if(fabs(GetDistance(position->globalPoint, target)) >= radius * 10){  
                std::vector<Point> path;
                if(GetDistance(position->globalPoint, target) > 50){
                    /*= Step1.1 : 计算轨迹与圆的交点 =*/
                     Point target_pt = CalcuTargetPoint(_rrt_long, target, radius);
                     target_pt.angle = 0;
                    /*= Step1.2 : RRT规划器获取轨迹 =*/
                    printf("rrt2_start\n");   
                    path = _rrt_long.optimal_rrt_planning({position->globalPoint.x, position->globalPoint.y, 0}, (Point)(target_pt), 4);  
                    // for(auto point : path){
                    //     printf("{x:%lf , y:%lf}\n", point.x, point.y);
                    // }
                    printf("x0 : %lf, y0 : %lf \n", gps_x, gps_y);
                    printf("x : %lf, y : %lf \n", target_pt.x, target_pt.y);
                    printf("rrt2_end\n");
                }else{
                    /*= Step1.1 : 计算轨迹与圆的交点 =*/
                    Point target_pt = CalcuTargetPoint(_rrt_short, target, radius);
                     target_pt.angle = 0;
                    /*= Step1.2 : RRT规划器获取轨迹 =*/
                    printf("rrt2_start\n");   
                    path = _rrt_short.optimal_rrt_planning({position->globalPoint.x, position->globalPoint.y, 0}, (Point)(target_pt), 4);  
                    // for(auto point : path){
                    //     printf("{x:%lf , y:%lf}\n", point.x, point.y);
                    // }
                    printf("x0 : %lf, y0 : %lf \n", gps_x, gps_y);
                    printf("x : %lf, y : %lf \n", target_pt.x, target_pt.y);
                    printf("rrt2_end\n");
                }
               
                /*= Step1.3 : 规划路径跟踪 =*/
                PathMove(path, goal_pt_track_time, maxSpeed, maxtime_ms, move_gaptime, fwd, back);

            }
            
            /*= Step2 :执行最后一步动作 =*/

            // 开启半吸环线程
            manual = false;
            ring_convey_spin = true;
            reinforce_stop = false;
            half_ring_get = true;
            turnToTarget(target,  maxSpeed * 0.5, 1000,  fwd, back);
            setStop(hold);
            moveToTarget(target, maxSpeed * 1, 1000, move_gaptime, fwd);
           // simpleMove(50, 0, 0.1, 10);
            setStop(hold);
            // 时间用于套环
            this_thread::sleep_for(200);
        }

        // 取桩
        void MoveForStake(Point target, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 15000, T move_gaptime = 10 ,int fwd = 1, int back = 1){
            RRT _rrt_long = *rrt_long; 
            RRT _rrt_short = *rrt_short;      
           /*= Step1 : 如果机器人在操作半径外则需要先到达可执行的操作半径范围内 =*/
            const T radius = 12;

            if(fabs(GetDistance(position->globalPoint, target)) >= radius*10){  
                std::vector<Point> path;
                if(GetDistance(position->globalPoint, target) > 65){
                    /*= Step1.1 : 计算轨迹与圆的交点 =*/
                     Point target_pt = CalcuTargetPoint(_rrt_long, target, radius);
                     target_pt.angle = 0;
                    /*= Step1.2 : RRT规划器获取轨迹 =*/
                    printf("rrt2_start\n");   
                    path = _rrt_long.optimal_rrt_planning({position->globalPoint.x, position->globalPoint.y, 0}, (Point)(target_pt), 4);  
                    // for(auto point : path){
                    //     printf("{x:%lf , y:%lf}\n", point.x, point.y);
                    // }
                    printf("rrt2_end\n");
                }else{
                    /*= Step1.1 : 计算轨迹与圆的交点 =*/
                    Point target_pt = CalcuTargetPoint(_rrt_short, target, radius);
                     target_pt.angle = 0;
                    /*= Step1.2 : RRT规划器获取轨迹 =*/
                    printf("rrt2_start\n");   
                    path = _rrt_short.optimal_rrt_planning({position->globalPoint.x, position->globalPoint.y, 0}, (Point)(target_pt), 4);  
                    // for(auto point : path){
                    //     printf("{x:%lf , y:%lf}\n", point.x, point.y);
                    // }
                    printf("rrt2_end\n");
                }
               
                /*= Step1.3 : 规划路径跟踪 =*/
                PathMove(path, goal_pt_track_time, maxSpeed, maxtime_ms, move_gaptime, fwd, back);
            }
            
            /*= Step2 :执行最后一步动作 =*/
            turnToTarget(target,  maxSpeed * 0.65, 1100,  fwd, back);
            moveToTarget(target, maxSpeed * 1, 1200, 10, fwd);
            gas_hold.state(100, pct);
            this_thread::sleep_for(300);
        }

        // 将桩放置到得分区(这里由于实际上得分区在四角，不考虑一般情况，只对特定的四角做讨论处理即可)
        void LayDownStake(Point target, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 15000, T move_gaptime = 10 ,int fwd = 1, int back = 1){
            RRT _rrt_long = *rrt_long; 
            RRT _rrt_short = *rrt_short;                                                                                                                                                  

            /*= Step1 : 如果机器人在操作半径外则需要先到达可执行的操作半径范围内 =*/
            const T radius = 50; 
            const T target_ = 62;
            const T step_back = 45;
            const T corner = 72;
            if(fabs(GetDistance(position->globalPoint, target)) >= radius){  
                std::vector<Point> path;
                if(GetDistance(position->globalPoint, target) > 70){
                    /*= Step1.1 : 计算轨迹与圆的交点 =*/
                     Point target_pt = CalcuTargetPoint(_rrt_long, target, radius);
                     target_pt.angle = 0;
                    /*= Step1.2 : RRT规划器获取轨迹 =*/
                    printf("rrt2_start\n");   
                    path = _rrt_long.optimal_rrt_planning({position->globalPoint.x, position->globalPoint.y, 0}, (Point)(target_pt), 4);  
                    // for(auto point : path){
                    //     printf("{x:%lf , y:%lf}\n", point.x, point.y);
                    // }
                    printf("rrt2_end\n");
                }else{
                    /*= Step1.1 : 计算轨迹与圆的交点 =*/
                    Point target_pt = CalcuTargetPoint(_rrt_short, target, radius);
                     target_pt.angle = 0;
                    /*= Step1.2 : RRT规划器获取轨迹 =*/
                    printf("rrt2_start\n");   
                    path = _rrt_short.optimal_rrt_planning({position->globalPoint.x, position->globalPoint.y, 0}, (Point)(target_pt), 4);  
                    // for(auto point : path){
                    //     printf("{x:%lf , y:%lf}\n", point.x, point.y);
                    // }
                    printf("rrt2_end\n");
                }
               
                /*= Step1.3 : 规划路径跟踪 =*/
                PathMove(path, goal_pt_track_time, maxSpeed, maxtime_ms, move_gaptime, fwd, back);

                /*= Step2 : 放桩 : 正常只需要对齐目标位置，把桩推进去即可放下再出来即可 =*/
                if(target.x == target_ && target.y == target_){
                    turnToTarget({corner, corner},  maxSpeed, 1500,  fwd, back);
                    moveToTarget({target_, target_}, maxSpeed, 1500, move_gaptime, fwd);
                    gas_hold.state(0, pct);
                    task::sleep(300);
                    moveToTarget({step_back, step_back}, maxSpeed, 1500, move_gaptime, fwd);
                }else if(target.x == -target_ && target.y == target_){
                    turnToTarget({-corner, corner},  maxSpeed, 1500,  fwd, back);
                    moveToTarget({-target_, target_}, maxSpeed, 1500, move_gaptime, fwd);
                    gas_hold.state(0, pct);
                    task::sleep(300);
                    moveToTarget({-step_back, step_back}, maxSpeed, 1500, move_gaptime, fwd);
                }else if(target.x == target_ && target.y == -target_){
                    moveToTarget({-step_back, -step_back}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                    turnToAngle(-45, maxSpeed, maxtime_ms, fwd, back);
                    simpleMove(100, 180, 1, 10);
                    gas_hold.state(0, pct);
                    task::sleep(300);
                    moveToTarget({-step_back, -step_back}, maxSpeed, 1500, move_gaptime, fwd);
                }else{
                    moveToTarget({-step_back, -step_back}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                    turnToAngle(-135, 70, maxtime_ms, fwd);
                    simpleMove(100, 180, 0.7, 10);
                    gas_hold.state(0, pct);
                    task::sleep(300);
                    moveToTarget({-step_back, -step_back}, maxSpeed, 1500, move_gaptime, fwd);
                }

            }else{      // 如果一开始就在半径内，则需要先退后出来，再转向、放桩
                    if(target.x == target_ && target.y == target_){
                        moveToTarget({step_back, step_back}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                        turnToTarget({corner, corner},  maxSpeed, maxtime_ms,  fwd, back);
                        manual = true;
                        reinforce_stop = true;
                        moveToTarget({target_, target_}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                        gas_hold.state(0, pct);
                        task::sleep(300);
                        moveToTarget({step_back, step_back}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                    }else if(target.x == -target_ && target.y == target_){
                        moveToTarget({-step_back, step_back}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                        turnToTarget({-corner, corner},  maxSpeed, maxtime_ms,  fwd, back);
                        manual = true;
                        reinforce_stop = true;
                        moveToTarget({-target_, target_}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                        gas_hold.state(0, pct);
                        task::sleep(300);
                        moveToTarget({-step_back, step_back}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                    }else if(target.x == target_ && target.y == -target_){
                        simpleMove(100, 180, 0.75, 10);
                        turnToAngle(-45, 80, 800, fwd);
                        reinforce_stop = true;
                        ring_color = 0;
                        ring_convey_spin = false;
                        photo_flag = false;
                        task::sleep(300);
                        moveToTarget({65, -65}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                        gas_hold.state(0, pct);
                        simpleMove(90, 180, 0.2, 10);
                    }else{
                        moveToTarget({-step_back, -step_back}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                        turnToAngle(45, 80, 1200, fwd);
                        reinforce_stop = true;
                        ring_color = 0;
                        ring_convey_spin = false;
                        photo_flag = false;
                        task::sleep(300);
                        moveToTarget({-target_, -target_}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                        gas_hold.state(0, pct);
                        turnToAngle(45, 60, 800, fwd);
                        simpleMove(70, 0, 0.55, 10);
                        //moveToTarget({-step_back, -step_back}, maxSpeed, maxtime_ms, move_gaptime, fwd);
                    }
            }
            

        }

        // 将环扣到固定桩(这里由于实际上得分区在四边，不考虑一般情况，只对特定的四边做讨论处理即可)
        void SlamDownRing(Point target, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 15000, T move_gaptime = 10 ,int fwd = 1, int back = 1){
            RRT _rrt_long = *rrt_long; 
            RRT _rrt_short = *rrt_short;       
            /*= Step1 : 如果机器人在操作半径外则需要先到达可执行的操作半径范围内 =*/
            const T head = 53; 
            const T head2 = 42;
            const T target_ = 60;
            const T side = 72;
            
            Point target_pt;
            Point target_pt2;
            /*= Step1.1 : 计算前视点轨迹 =*/
            if(target.x == 0 && target.y == target_){
                target_pt = {-1.8, head};
                target_pt2 = {-1.8, head2};
                std::vector<Point> path = _rrt_short.optimal_rrt_planning({gps_x, gps_y}, (Point)(target_pt2), 2);  
                // 先正向对齐
                //turnToAngle(-90, maxSpeed, maxtime_ms);
                moveToTarget(target_pt2, maxSpeed, maxtime_ms, move_gaptime);
                task::sleep(300);
                // 先正向对齐
                turnToAngle(0, maxSpeed, maxtime_ms);
                setStop(hold);
                // 对齐侧向
                VisionSensorMove(154, maxSpeed, maxtime_ms, 0);
                task::sleep(300);
                // 转向桩
                turnToAngle(180, maxSpeed, maxtime_ms);
                // 对齐正向
                DistanceSensorMove(340, maxSpeed * 0.4, 5000, 0);
                setStop(hold);
                // 正向对齐固定桩
                task::sleep(300);
                // 转向桩
                turnToAngle(180,  maxSpeed, maxtime_ms);
                setStop(hold);
                thread arm_lift(Arm_Lift);
                thread arm_down(Arm_Down);
                task::sleep(2000);
            }else if(target.x == target_ && target.y == 0){

                target_pt = {head, -3};
                target_pt2 = {head2, -3};
                std::vector<Point> path = _rrt_short.optimal_rrt_planning({gps_x, gps_y}, (Point)(target_pt2), 2);  
                // 先正向对齐
                //turnToAngle(-90, maxSpeed, maxtime_ms);
                moveToTarget(target_pt, maxSpeed, maxtime_ms, move_gaptime);
                task::sleep(300);
                // 先正向对齐
                turnToAngle(90, maxSpeed, maxtime_ms);
                setStop(hold);

                // 对齐侧向
                VisionSensorMove(160, maxSpeed, maxtime_ms, 0);
                task::sleep(300);
                // 转向桩
                turnToAngle(-90, maxSpeed, maxtime_ms);
                // 对齐正向
                DistanceSensorMove(140, maxSpeed * 0.4, 5000, 0);
                setStop(hold);
                // 正向对齐固定桩
                task::sleep(300);
                // 转向桩
                turnToAngle(-90,  maxSpeed, maxtime_ms);
                setStop(hold);
                // 放环
                convey_belt.spin(forward,70,pct);
                task::sleep(700);
                convey_belt.spin(forward,0,pct);
            }else if(target.x == 0 && target.y == -target_){
                target_pt = {-1.8, -head};
                target_pt2 = {-1.8, -head2};
                std::vector<Point> path = _rrt_short.optimal_rrt_planning({gps_x, gps_y}, (Point)(target_pt2), 2);  
                // 先正向对齐
                //turnToAngle(-90, maxSpeed, maxtime_ms);
                moveToTarget(target_pt2, maxSpeed, maxtime_ms, move_gaptime);
                task::sleep(300);
                // 先正向对齐
                turnToAngle(180, maxSpeed, maxtime_ms);
                setStop(hold);
                // 对齐侧向
                VisionSensorMove(154, maxSpeed, maxtime_ms, 0);
                task::sleep(300);
                // 转向桩
                turnToAngle(0, maxSpeed, maxtime_ms);
                // 对齐正向
                DistanceSensorMove(340, maxSpeed * 0.4, 5000, 0);
                setStop(hold);
                // 正向对齐固定桩
                task::sleep(300);
                // 转向桩
                turnToAngle(0,  maxSpeed, maxtime_ms);
                setStop(hold);
                // 放环
                thread arm_lift(Arm_Lift);
                thread arm_down(Arm_Down);
                task::sleep(2500);  
            }else{

                target_pt = {-head, -3};
                target_pt2 = {-head2, -3};
                std::vector<Point> path = _rrt_short.optimal_rrt_planning({gps_x, gps_y}, (Point)(target_pt2), 2);  
                // 先正向对齐
                //turnToAngle(-90, maxSpeed, maxtime_ms);
                moveToTarget(target_pt, maxSpeed, maxtime_ms, move_gaptime);
                task::sleep(300);
                // 先正向对齐
                turnToAngle(-90, maxSpeed, maxtime_ms);
                setStop(hold);

                // 对齐侧向
                VisionSensorMove(165, maxSpeed, maxtime_ms, 0);
                task::sleep(300);
                // 转向桩
                turnToAngle(90, maxSpeed, maxtime_ms);
                // 对齐正向
                DistanceSensorMove(140, maxSpeed * 0.4, 5000, 0);
                setStop(hold);
                // 正向对齐固定桩
                task::sleep(300);
                // 转向桩
                turnToAngle(96,  maxSpeed, maxtime_ms);
                setStop(hold);
                // 放环
                convey_belt.spin(forward,70,pct);
                task::sleep(700);
                convey_belt.spin(forward,0,pct);
            }
                
        }

        // 借助一次粗规划求解目标点
        Point CalcuTargetPoint(RRT _rrt, Point target, T radius){
            // 误差允值是2倍步长
            T error = 2 * _rrt.step;

            // 先进行一次粗规划
            Point start_pt = {position->globalPoint.x, position->globalPoint.y};
            std::vector<Point> path = _rrt.rrt_planning(start_pt, (Point)(target));  // 这里一定要强制类型转换为Point
            for(auto point : path){
                    printf("{x:%lf , y:%lf}\n", point.x, point.y);
            }
            // 二分法计算坐标
            int left = 0;
            int right = path.size() - 1;
            T distance = -1;
            while(fabs(distance - radius) >= error){

                int middle = (left + right) / 2;
                
                distance = sqrt((path[middle].x - target.x) * (path[middle].x - target.x) + (path[middle].y - target.y) * (path[middle].y - target.y));
                
                printf("left:%d right:%d middle:%d distance : %lf \n",left, right, middle, distance);
                
                // 计算距离
                if(distance < radius){
                    // 从期望上讲，如果能够满足单调关系，则新middle会比老middle更大，否则会陷入死循环
                    right = middle;
                    // 对死循环困境进行处理，如果出现困境则强制right++变得更大
                    T dis_nxt = sqrt((path[(left + right) / 2].x - target.x) * (path[(left + right) / 2].x - target.x) + (path[(left + right) / 2].y - target.y) * (path[(left + right) / 2].y - target.y));
                    while(dis_nxt < distance){
                        right++;
                        dis_nxt = sqrt((path[(left + right) / 2].x - target.x) * (path[(left + right) / 2].x - target.x) + (path[(left + right) / 2].y - target.y) * (path[(left + right) / 2].y - target.y));
                    }
                    
                }else{
                    // 从期望上讲，如果能够满足单调关系，则新middle会比老middle更小，否则会陷入死循环
                    left = middle;
                    // 对死循环困境进行处理，如果出现困境则强制left--变得更小
                    T dis_nxt = sqrt((path[(left + right) / 2].x - target.x) * (path[(left + right) / 2].x - target.x) + (path[(left + right) / 2].y - target.y) * (path[(left + right) / 2].y - target.y));
                    while(dis_nxt > distance){
                        left--;
                        dis_nxt = sqrt((path[(left + right) / 2].x - target.x) * (path[(left + right) / 2].x - target.x) + (path[(left + right) / 2].y - target.y) * (path[(left + right) / 2].y - target.y));
                    }
                }
            }

            return path[(left + right) / 2];
            
        }

    };
}