#pragma once
#include "vex.h"
#include "tjulib.h"

namespace tjulib{

    using namespace vex;
    typedef double T;

    class RemoteDebug{
    protected:
        Position *position = NULL;
    private:
        Point globalPoint{0,0,0}; //x, y, angle
    public:
        RemoteDebug(Position *position) : position(position) {};
        
        int PositionDebugSerial(){

            while(1){

                // 获取当前坐标
                globalPoint = position->globalPoint;
                globalPoint.angle = imu.rotation();
                while(globalPoint.angle>360){ globalPoint.angle -= 360; }
                while(globalPoint.angle<0){ globalPoint.angle += 360; }
             //   printf("------------DashBoardPos------------\n");

                
                // 将坐标输出
             //   printf("(%.2f,%.2f,%.2f)\n", globalPoint.x, globalPoint.y, globalPoint.angle);
                
                task::sleep(50);
            }
            return 0;
        }

    };
    

};   