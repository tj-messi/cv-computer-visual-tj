#pragma once
#include "Math-Functions.h"
#include "vex.h"
#include "GPS.hpp"

namespace tjulib{

    using namespace vex;
    typedef double T;


    // 定位策略接口
    class Position {
    public:
        //全局坐标系下的当前位置，包括 x 和 y 坐标以及朝向角度
        Point globalPoint{0,0,0}; //x, y, angle(in degree)
        Point prevGlobalPoint{0,0,0}; 
        T LeftBaseDistance =0, RightBaseDistance = 0;

    public:
        virtual ~Position() {}
        virtual void executePosition() = 0;
        virtual void setPosition(float newX, float newY, float newAngle) = 0;
    };


};