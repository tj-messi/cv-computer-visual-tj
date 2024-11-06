#pragma once

#include "vex.h"

typedef double T;

namespace tjulib
{
    struct pidParams
    {
        T kp, ki, kd;         // pid系数
        T integralActiveZone; // 积分发挥作用区域
        T errorThreshold;     // 误差允值
        T minSpeed;           // 最小速度，要根据车的重量以及pid参数设置
        int stop_num;         // 回震限数
        pidParams(T kp, T ki, T kd, T integralActiveZone, T errorThreshold, T minSpeed, int stop_num)
            : kp(kp), ki(ki), kd(kd), integralActiveZone(integralActiveZone), errorThreshold(errorThreshold), minSpeed(minSpeed), stop_num(stop_num){};
    };

    class pidControl
    {
    private:
        double error = 0;      // Distance from target forward distance
        double lastError = 0;  // Keep track of last error for the derivative (rate of change)
        double integral = 0;   // Integral accumulates the error, speeding up if target is not reached fast enough
        double derivative = 0; // Derivative smooths out oscillations and counters sudden changes
        
    public:
        pidParams *params = NULL;
        int cnt = 0;   // 回震计数

    public:
        
        pidControl(pidParams *params) : params(params){};

        T pidCalcu(T target, T maxSpeed, T feedback = 0);

        bool overflag();
        
        void reset();

        void resetpid();
    };

}