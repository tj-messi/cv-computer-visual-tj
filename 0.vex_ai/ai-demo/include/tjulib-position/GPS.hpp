#pragma once
#include "Math-Functions.h"
#include "vex.h"

namespace tjulib{

    using namespace vex;
    typedef double T;
    class GPS {
    public:
        gps &GPS_;
        const T offset_x;
        const T offset_y;
        const T PI = 3.1415926535;
        GPS(gps &GPS_, T offset_x, T offset_y):
           GPS_(GPS_), offset_x(offset_x), offset_y(offset_y) {}

        T gpsX() {
            T theta = GPS_.heading(deg) / 180 * PI;  // 弧度制
            return   GPS_.xPosition(inches) - ( offset_y * sin(theta) + offset_x * cos(theta) );
        }

        T gpsY() {
            T theta = GPS_.heading(deg)/ 180 * PI;  // 弧度制
            return   GPS_.yPosition(inches) - ( offset_y * cos(theta) - offset_x * sin(theta) );
        }
        
        T gpsHeading(){
            return GPS_.heading(deg);
        }


    };
}