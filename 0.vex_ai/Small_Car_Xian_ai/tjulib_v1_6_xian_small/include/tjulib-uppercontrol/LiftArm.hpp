#pragma once
#include "vex.h"
#include <string>
#include <iostream>
using namespace vex;
typedef double T;

namespace tjulib
{
    int Arm_Lift(){
        lift_arm.spin(forward); // 电机正转
        task::sleep(1200);
        lift_arm.setStopping(brakeType::hold);
        lift_arm.stop(hold);
        // 放环
        convey_belt.spin(reverse,75,pct);
        return 0;
    }

    int Arm_Down(){     //arm_lift和arm_down是一起配合使用的
        task::sleep(3000);
        convey_belt.spin(forward,0,pct);
        lift_arm.setStopping(brakeType::coast);
        lift_arm.spin(reverse, 0, pct); // 电机反转

        lift_arm.stop(hold);
        return 0;
    }
}