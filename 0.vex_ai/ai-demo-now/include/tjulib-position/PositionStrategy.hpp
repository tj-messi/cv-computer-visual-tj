#pragma once
#include <iostream>
#include <memory>
#include "Math-Functions.h"
#include "difOdom.hpp"
#include "Odom.hpp"
#include "vex.h"

namespace tjulib{

    using namespace vex;
    typedef double T;

    // 策略模式：上下文类(选择制定的策略模式)
    class Context{
    public:
        Position* position = NULL;
    public:
        Context(Position *position) : position(position)  {}

        int startPosition() {
            position->executePosition();

            return 0;
        }

        void setPosition(Point point){
            position->setPosition(point.x, point.y, point.angle);
        }

        // int startPosition_odom() {
        //     odom->executePosition();

        //     return 0;
        // }

        // void setPosition_odom(Point point){
        //     odom->setPosition(point.x, point.y, point.angle);
        // }
    };


};