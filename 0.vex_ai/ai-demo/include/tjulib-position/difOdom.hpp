#pragma once
#include "Math-Functions.h"
#include "vex.h"
#include "Position.hpp"

namespace tjulib{

    using namespace vex;
    typedef double T;

    class Dif_Odom : public Position{

    private:
        
        Math OMath;
        inertial& imu;
        std::vector<vex::motor*> &_leftMotors;
        std::vector<vex::motor*> &_rightMotors;

        Point localDeltaPoint{0,0,0};

        T r_motor = 5;

        void resetTotalDistance(){
            for(motor* m : _leftMotors)
                m->resetPosition();
            for(motor* m : _rightMotors)
                m->resetPosition();
        }

    public:
        Dif_Odom(std::vector<vex::motor*> &leftMotors, std::vector<vex::motor*> &rightMotors, T wheelCircumference, T r_motor,                 
                inertial& imu
                ) : _leftMotors(leftMotors), _rightMotors(rightMotors),
                OMath(wheelCircumference), r_motor(r_motor), imu(imu)
        {
            resetTotalDistance();
        }


        void OdomRun(){
            T LeftSideMotorPosition=0, RightSideMotorPosition = 0;
            T LastLeftSideMotorPosition=0, LastRightSideMotorPosition = 0;
            
            while(true) {         
                // 获取imu读数
                globalPoint.angle = OMath.getRadians(imu.rotation());
                T DeltaHeading = globalPoint.angle - prevGlobalPoint.angle;
                // 读取电机转角
                LeftSideMotorPosition = 0;
                RightSideMotorPosition = 0;
                for(motor* m : _leftMotors) LeftSideMotorPosition += OMath.getRadians(m->position(deg)) / _leftMotors.size();
                for(motor* m : _rightMotors) RightSideMotorPosition += OMath.getRadians(m->position(deg)) / _rightMotors.size();
                
                T LeftSideDistance=(LeftSideMotorPosition - LastLeftSideMotorPosition); 
                T RightSideDistance=(RightSideMotorPosition-LastRightSideMotorPosition);
                LeftBaseDistance = LeftSideMotorPosition * r_motor; RightBaseDistance = RightSideMotorPosition * r_motor;

                // 计算相对坐标系中的变化
                T CenterRidus = DeltaHeading ? (LeftSideDistance + RightSideDistance)/(2*DeltaHeading) : MAXFLOAT;
                localDeltaPoint.x = CenterRidus * (1-cos(DeltaHeading));
                localDeltaPoint.y = CenterRidus * sin(DeltaHeading);

                // 计算全局坐标中的变化
                globalPoint.x = prevGlobalPoint.x - localDeltaPoint.x * cos(prevGlobalPoint.angle) + localDeltaPoint.y * sin(prevGlobalPoint.angle);
                globalPoint.y = prevGlobalPoint.y + localDeltaPoint.x * sin(prevGlobalPoint.angle) + localDeltaPoint.y * cos(prevGlobalPoint.angle);
                // 更新坐标
                prevGlobalPoint.y = globalPoint.y;
                prevGlobalPoint.x = globalPoint.x;
                prevGlobalPoint.angle = globalPoint.angle;
                LastLeftSideMotorPosition = LeftSideMotorPosition;
                LastRightSideMotorPosition = RightSideMotorPosition;     
                // 调试信息打印
                printf("x : %lf cell , y : %lf cell , heading : %lf imu : %lf\n",globalPoint.x / cell ,globalPoint.y / cell , globalPoint.angle, imu.angle());
               // printf("GPS x : %lf cell , y : %lf cell , angle : %lf \n", GPS.xPosition() , GPS.yPosition(), GPS.heading());
                this_thread::sleep_for(10);
            }

        }
        void setPosition(float newX, float newY, float newAngle = -114514) override{

            prevGlobalPoint.x = newX;
            prevGlobalPoint.y = newY;
            
            globalPoint.x = newX;
            globalPoint.y = newY;
            if(fabs(newAngle - 114514)>1){
                imu.setRotation(newAngle, deg);
                imu.setHeading(newAngle, deg);
                globalPoint.angle = newAngle;
            }
            
        }
        void executePosition() override{
            OdomRun();
        }
    };

}