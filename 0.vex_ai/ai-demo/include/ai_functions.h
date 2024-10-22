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

// Moves the robot to a specified position and orientation
void moveToPosition(double target_x, double target_y, double target_theta);

// Finds a target object based on the specified type
DETECTION_OBJECT findTarget(int type);

// Drives to the closest specified object
void goToObject(OBJECT type);

// Turns the robot to a specific angle with given tolerance and speed
void turnTo(double angle, int tolerance, int speed);

// Drives the robot in a specified heading for a given distance and speed
void driveFor(int heading, double distance, int speed);

// Grabs a ring when the arm is positioned over it
void grabRing();

// Drops a ring on a goal when the arm is positioned over the goal
void dropRing();
