#pragma once
// 启用手动PID则打开，不启用则直接注释掉
//#define ManualPID
#include "vex.h"
#include "basechassis.hpp"
#include "tjulib-position/PositionStrategy.hpp"
#include "pidControl.hpp"
#include <cmath>
extern double zero_drift_error;
namespace tjulib
{
    using namespace vex;

    class ChassisMotionFunc{
    protected:
        Position *position = NULL;
        const T PI = 3.14159265358979323846;
        const T toDegrees = 180.0 / PI; // ���ȳ��Ը�����תΪ�Ƕ�
        const T r_motor = 0;
    protected:

        // 计算转向最优误差角
        T optimalTurnAngle(T targetAngle, T currentAngle){
            return fabs(targetAngle - currentAngle) <= 180 ?
                        targetAngle - currentAngle :
                        (targetAngle - currentAngle) >= 0 ? (targetAngle - currentAngle) - 360 : (targetAngle - currentAngle) + 360;
        }

        // 获得两点距离
        T GetDistance(Point p1, Point p2){
            return sqrt((p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y));
        }
        
        // 获得当前位置坐标和目标的距离
        //差速定位法
        T GetDistance(Point target) {
            Point car = position->globalPoint;
            return GetDistance(car, target);
        }

        // 获得两点之间的夹角
        T getDegree(Point target)
        {
            T relativeX = target.x - position->globalPoint.x;
            T relativeY = target.y - position->globalPoint.y;

            T deg = toDegrees * atan2(relativeY, relativeX);
            
            if((relativeX > 0 && relativeY > 0) || (relativeX < 0 && relativeY > 0))
                deg += 0;
            else if((relativeX > 0 && relativeY < 0) || (relativeX < 0 && relativeY < 0))
                deg += 360;
            else
                ;
            return deg;
        }

        T getDegree(Point pt_1, Point pt_2)
        {
            T relativeX = pt_2.x - pt_1.x;
            T relativeY = pt_2.y - pt_1.y;

            T deg = toDegrees * atan2(relativeY, relativeX);
            
            if((relativeX > 0 && relativeY > 0) || (relativeX < 0 && relativeY > 0))
                deg += 0;
            else if((relativeX > 0 && relativeY < 0) || (relativeX < 0 && relativeY < 0))
                deg += 360;
            else
                ;
            return deg;
        }

        // 获得机器人自己的相对坐标系下航向角（与定位target结合使用）
        T getLocalDegree(Point target){
            // 计算两点连线与y轴正方向之间的夹角(0~360°)
            T deg = 90 - getDegree(target);
            if (deg < 0)
                deg += 360;
            // 转换到相对坐标系下，由于position存的是弧度制，因此要转换成角度值
            deg = deg  - position->globalPoint.angle/ PI * 180;
            if (deg < 0)
                deg += 360;
            return deg;
        }


    public:
        ChassisMotionFunc(Position *_position, const T _r_motor)
            : position(_position), r_motor(_r_motor){};
    };


    class BaseChassis : public ChassisMotionFunc{
    private:
        T current_yaw = 0;
        T initial_yaw = 0;
        
    protected:
        vex::brakeType btype = vex::brakeType::brake;       // 停止类型这里默认设置为的是brake
        std::vector<std::vector<vex::motor*>*> &chassisMotors;
        pidControl *motorpidControl = NULL; // 轮子电机pid控制器

    public: 
        
        BaseChassis(std::vector<std::vector<vex::motor*>*> &_chassisMotors, pidControl *_motorpidControl, Position *_position, T _r_motor)
            : ChassisMotionFunc(_position, _r_motor), chassisMotors(_chassisMotors),  motorpidControl(_motorpidControl){}

        virtual ~BaseChassis() {}


        // pid控制的轮子定速旋转
        virtual void motorspin(motor *motor, T speed, T maxSpeed, T respondTime = 100, T gaptime = 20) = 0;

        // 设置停止类型 : brake coast hold
        virtual void setBrakeType(vex::brakeType brake_type) = 0;
        
        // 急刹 : 与setBrakeType一起配合使用
        virtual void setStop(vex::brakeType type) = 0;

        // 无PID版本手动控制
        virtual void ManualDrive_nonPID() = 0;

        // PID版本手动控制
        virtual void ManualDrive_PID() = 0;

    };

} // namespace tjulib
