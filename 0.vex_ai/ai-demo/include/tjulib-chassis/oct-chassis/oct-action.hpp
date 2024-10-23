#include "vex.h"
#include "tjulib-uppercontrol/ConveyRing.h"
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

        RRT *rrt = NULL;
        Oct_Action(std::vector<std::vector<vex::motor*>*>& _chassisMotors, pidControl* _motorpidControl, Position* _position, const T _r_motor, pidControl* _fwdpid, pidControl* _turnpid, PurePursuit *_ppcontrol, RRT *rrt) :
            Oct_BaseChassis(_chassisMotors, _motorpidControl, _position, _r_motor), 
            Oct_StraChassis(_chassisMotors, _motorpidControl, _position, _r_motor, _fwdpid, _turnpid),
            Oct_CurChassis(_chassisMotors, _motorpidControl, _position, _r_motor, _fwdpid, _turnpid, _ppcontrol),
            rrt(rrt) {}
        

        void ActionChoose(int action_index, Point target, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 15000, T move_gaptime = 10 ,int fwd = 1, int back = 0){
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
            default:
                break;
            }
           
        }


        // 取环
        void MoveForRing(Point target, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 15000, T move_gaptime = 10 ,int fwd = 1, int back = 0){
            // 先通过粗规划获取目标点
            T radius = 8;
            Point target_pt = CalcuTargetPoint(target, radius);

            printf("rrt_start\n");
            std::vector<Point> path = rrt->optimal_rrt_planning({position->globalPoint.x, position->globalPoint.y}, (Point)(target_pt), 4);  // 这里一定要强制类型转换为Point
            // for(auto point : path){
            //     printf("{x:%lf , y:%lf}\n", point.x, point.y);
            // }
            printf("rrt_end\n");
            
            // 目标点追踪
            PathMove(path, goal_pt_track_time, maxSpeed, maxtime_ms, move_gaptime, fwd, back);
            


        }

        // 取桩
        void MoveForStake(Point target, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 15000, T move_gaptime = 10 ,int fwd = 1, int back = 1){

        }

        // 将桩放置到得分区
        void LayDownStake(Point target, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 15000, T move_gaptime = 10 ,int fwd = 1, int back = 1){

        }

        // 将环扣到固定桩
        void SlamDownRing(Point target, T goal_pt_track_time = 100, T maxSpeed = 100, T maxtime_ms = 15000, T move_gaptime = 10 ,int fwd = 1, int back = 1){

        }

        // 借助一次粗规划求解目标点
        Point CalcuTargetPoint(Point target, T radius){
            // 误差允值是2倍不常
            T error = 2 * rrt->step;

            // 先进行一次粗规划
            Point start_pt = {position->globalPoint.x, position->globalPoint.y};
            std::vector<Point> path = rrt->rrt_planning(start_pt, (Point)(target));  // 这里一定要强制类型转换为Point

            // 二分法计算坐标
            int left = 0;
            int right = path.size() - 1;
            T distance = -1;
            while(fabs(distance - radius) >= error){
                int middle = (left + right) / 2;
                distance = sqrt((path[middle].x - target.x) * (path[middle].x - target.x) + (path[middle].y - target.y) * (path[middle].y - target.y));
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
                    while(dis_nxt < distance){
                        left--;
                        dis_nxt = sqrt((path[(left + right) / 2].x - target.x) * (path[(left + right) / 2].x - target.x) + (path[(left + right) / 2].y - target.y) * (path[(left + right) / 2].y - target.y));
                    }
                }
            }

            return path[(left + right) / 2];
            
        }

    };
}