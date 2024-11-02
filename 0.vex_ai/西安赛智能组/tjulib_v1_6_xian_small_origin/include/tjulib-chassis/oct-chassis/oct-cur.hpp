#pragma once    
#include "tjulib-chassis/oct-chassis/oct-straight.hpp"
#include "tjulib-motionplanner/tjulib-actionplanner/purepursuit.hpp"

extern double zero_drift_error;

namespace tjulib
{
    using namespace vex;

    class Oct_CurChassis : virtual public Oct_StraChassis {

    private:
        const double PI = 3.14159265358979323846;

    public:
        PurePursuit *ppcontrol = NULL;

        Oct_CurChassis(std::vector<std::vector<vex::motor*>*>& _chassisMotors, pidControl* _motorpidControl, Position* _position, const T _r_motor, pidControl* _fwdpid, pidControl* _turnpid, PurePursuit *_ppcontrol) :
            Oct_BaseChassis(_chassisMotors, _motorpidControl, _position, _r_motor), Oct_StraChassis(_chassisMotors, _motorpidControl, _position, _r_motor, _fwdpid, _turnpid),  ppcontrol(_ppcontrol){}

        /*===========轨迹跟踪移动(这里使用PurePursuit控制器进行轨迹跟踪) =============*/
        // 其中比较重要的调节参数包括 最大速度maxSpeed(0~100), 每次进行追踪前视点的时间goal_pt_tack_time
        void PathMove(std::vector<Point> path, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 10000, T move_gaptime = 10 ,int fwd = 1, int back = 0){
            Point goal_point;
            bool intersectFound = false;
            int LastFoundIndex = 0;
            const T lookAheadDis = 15;
            int cnt = 0;
            const T errorThredhold = 3;
            timer time;
            time.clear();
            const Point startPoint = position->globalPoint; // 一开始的时候车辆的初始位置
            while(cnt <= 15){
                if(time.time(msec)>=maxtime_ms){
                    break;
                }
                if(fabs(GetDistance(position->globalPoint, path[path.size() - 1])) <= errorThredhold){
                    cnt++;
                }
                if(LastFoundIndex == path.size() - 1){       // 最后一段直接直线移动到目标点即可
                    // 计算跟踪path[path.size() - 1]时候的夹角
                    // T targetDeg = getDegree(path[LastFoundIndex-1], path[LastFoundIndex]);
                    goal_point = path[LastFoundIndex];
                    T targetDeg = getDegree(position->globalPoint, path[LastFoundIndex]);
                    targetDeg = 90 - targetDeg;
                    if (targetDeg < 0){
                        targetDeg += 360;
                    }
                    // 调转车头方向
                    if(back){
                        targetDeg += 180;
                    }
                    targetDeg = Math::getWrap360(targetDeg);
                    goal_point.angle = targetDeg;

                    if(fabs(GetDistance(position->globalPoint, path[path.size() - 1])) <= 1.5 * errorThredhold){
                        goal_point.x = path[path.size() - 1].x;
                        goal_point.y = path[path.size() - 1].y;
                        goal_point = path[LastFoundIndex];
                        T targetDeg = getDegree(position->globalPoint, path[path.size() - 1]);
                        targetDeg = 90 - targetDeg;
                        if (targetDeg < 0){
                            targetDeg += 360;
                        }
                        // 调转车头方向
                        if(back){
                            targetDeg += 180;
                        }
                        goal_point.angle = targetDeg;
                    }

                    // 最后一段直接直线移动到目标点
                    if(targetDeg <= 60 || fabs(targetDeg - 360) <= 60){
                        //RotMoveToTarget(goal_point, maxSpeed , 1000, move_gaptime, fwd);
                        moveToTarget_forPP(goal_point, maxSpeed, move_gaptime, move_gaptime, fwd);
                    }else{
                        moveToTarget_forPP(goal_point, maxSpeed, move_gaptime, move_gaptime, fwd);
                    }
                    //printf("111111111111111\n");
                    break;
                }else{  // 不是最后一段，要跟踪目标点
                    // 获取前视跟踪点
                    std::pair<Point, bool> result = ppcontrol->goal_pt_search(path, lookAheadDis, LastFoundIndex);
                    goal_point = result.first;
                    intersectFound = result.second;
                    // 对前视跟踪点进行追踪
                    T targetDeg = getDegree(position->globalPoint, goal_point);
                    targetDeg = 90 - targetDeg;
                    // if(LastFoundIndex == 0){
                    //     targetDeg = getDegree(startPoint, path[0]);
                    // }else{
                    //     targetDeg = getDegree(path[LastFoundIndex - 1], path[LastFoundIndex]);
                    // }
                    if (targetDeg < 0){
                        targetDeg += 360;
                    }
                    
                    // 调转车头方向
                    if(back){
                        targetDeg += 180;
                    }
                    targetDeg = Math::getWrap360(targetDeg);
                    goal_point.angle = targetDeg;

                    // printf("t_x : %lf, t_y : %lf, t_angle : %lf \n", goal_point.x , goal_point.y, goal_point.angle);
                    if(fabs(GetDistance(position->globalPoint, path[path.size() - 1])) <= 5 * errorThredhold){
                        goal_point.x = path[path.size() - 1].x;
                        goal_point.y = path[path.size() - 1].y;
                        goal_point = path[LastFoundIndex];
                        T targetDeg = getDegree(position->globalPoint, path[path.size() - 1]);
                        targetDeg = 90 - targetDeg;
                        if (targetDeg < 0){
                            targetDeg += 360;
                        }
                        // 调转车头方向
                        if(back){
                            targetDeg += 180;
                        }
                        goal_point.angle = targetDeg;
                        // 最后一段直接直线移动到目标点
                        if(targetDeg <= 120 && fabs(targetDeg - 360) <= 120){
                            moveToTarget_forPP(goal_point, maxSpeed, move_gaptime, move_gaptime, fwd);
                            //RotMoveToTarget(goal_point, maxSpeed , move_gaptime, move_gaptime, fwd);
                        }else{
                            moveToTarget_forPP(goal_point, maxSpeed, move_gaptime, move_gaptime, fwd);
                            //RotMoveToTarget(goal_point, maxSpeed , 1000, move_gaptime, fwd);
                        }
                        break;
                    }else{
                        // 最后一段直接直线移动到目标点
                        if(targetDeg <= 120 && fabs(targetDeg - 360) <= 120){
                            moveToTarget_forPP(goal_point, maxSpeed, move_gaptime, move_gaptime, fwd);
                            //RotMoveToTarget(goal_point, maxSpeed , move_gaptime, move_gaptime, fwd);
                        }else{
                            moveToTarget_forPP(goal_point, maxSpeed, move_gaptime, move_gaptime, fwd);
                            //RotMoveToTarget(goal_point, maxSpeed , move_gaptime, move_gaptime, fwd);
                        }
                    }
                     
                    //
                }
                task::sleep(10);
            }
        }  
        
    };
}