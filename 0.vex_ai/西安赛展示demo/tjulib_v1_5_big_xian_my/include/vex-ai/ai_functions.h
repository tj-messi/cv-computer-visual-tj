/*----------------------------------------------------------------------------*/
/*                                                                            */
/*    Copyright (c) Innovation First 2023 All rights reserved.                */
/*    Licensed under the MIT license.                                         */
/*                                                                            */
/*    Module:     ai_functions.cpp                                            */
/*    Author:     VEX Robotics Inc.                                           */
/*    Created:    11 August 2023                                              */
/*    Description:  Header for AI robot movement functions                    */
/*                                                                            */
/*----------------------------------------------------------------------------*/

#include <vex.h>
#include <robot-config.h>

enum OBJECT {
    MobileGoal,
    RedRing,
    BlueRing,
    BothRings
};

using namespace vex;

// Calculates the distance to a given target (x, y)
double distanceTo(double target_x, double target_y);

// Finds a target object based on the specified type
DETECTION_OBJECT findTarget(int type);
