/*----------------------------------------------------------------------------*/
/*                                                                            */
/*    Copyright (c) Innovation First 2023 All rights reserved.                */
/*    Licensed under the MIT license.                                         */
/*                                                                            */
/*    Module:     ai_functions.cpp                                            */
/*    Author:     VEX Robotics Inc.                                           */
/*    Created:    11 August 2023                                              */
/*    Description:  Helper movement functions for VEX AI program              */
/*                                                                            */
/*----------------------------------------------------------------------------*/


#include "vex.h"
#include "ai_functions.h"
#include <string>
#include <iostream>
using namespace vex;
using namespace std;


// Calculates the distance to the coordinates from the current robot position
double distanceTo(double target_x, double target_y){
    double distance = sqrt(pow((target_x - GPS.xPosition(vex::distanceUnits::cm)), 2) + pow((target_y - GPS.yPosition(vex::distanceUnits::cm)), 2));
    return distance;
}

// Calculates the bearing to drive to the target coordinates in a straight line aligned with global coordinate/heading system.
double calculateBearing(double currX, double currY, double targetX, double targetY) {
    // Calculate the difference in coordinates
    double dx = targetX - currX;
    double dy = targetY - currY;

    // Calculate the bearing in radians
    double bearing_rad = atan2(dy, dx);

    // Convert to degrees
    double bearing_deg = bearing_rad * 180 / M_PI;

    // Normalize to the range 0 to 360
    if (bearing_deg < 0) {
        bearing_deg += 360;
    }

    // Convert from mathematical to navigation coordinates
    bearing_deg = fmod(90 - bearing_deg, 360);
    if (bearing_deg < 0) {
        bearing_deg += 360;
    }

    return bearing_deg;
}

// Turns the robot to face the angle specified, taking into account a tolerance and speed of turn.
void turnTo(double angle, int tolerance, int speed){
    double current_heading = GPS.heading();
    double angle_to_turn = angle - current_heading;

    // Normalize the angle to the range [-180, 180]
    while (angle_to_turn > 180) angle_to_turn -= 360;
    while (angle_to_turn < -180) angle_to_turn += 360;

    // Determine the direction to turn (left or right)
    turnType direction = angle_to_turn > 0 ? turnType::left : turnType::right;
    Drivetrain.turn(direction, speed, velocityUnits::pct);
    while (1) {
    
        current_heading = GPS.heading();
        // Check if the current heading is within a tolerance of degrees to the target
        if (current_heading > (angle - tolerance) && current_heading < (angle + tolerance)) {
            break;
        }

    }
    Drivetrain.stop();
}

// Moves the robot toward the target at the specificed heading, for a distance at a given speed.
void driveFor(int heading, double distance, int speed){
    // Determine the smallest degree of turn
    double angle_to_turn = heading - GPS.heading();
    while (angle_to_turn > 180) angle_to_turn -= 360;
    while (angle_to_turn < -180) angle_to_turn += 360;

    // Decide whether to move forward or backward
    // Allos for a 5 degree margin of error that defaults to forward
    directionType direction = fwd;
    if (std::abs(angle_to_turn) > 105) {
        angle_to_turn += angle_to_turn > 0 ? -180 : 180;
        direction = directionType::rev;
    } else if (std::abs(angle_to_turn) < 75) {
        angle_to_turn += angle_to_turn > 0 ? 180 : -180;
        direction = directionType::fwd;
    }

    Drivetrain.driveFor(direction, distance, vex::distanceUnits::cm, speed, velocityUnits::pct);
}

// Method that moves to a given (x,y) position and a desired target theta to finish movement facing
void moveToPosition(double target_x, double target_y, double target_theta = -1) {
    // Calculate the angle to turn to face the target
    double intialHeading = calculateBearing(GPS.xPosition(distanceUnits::cm), GPS.yPosition(distanceUnits::cm), target_x, target_y);
    // Turn to face the target
    turnTo(intialHeading, 10, 15);
    double distance = distanceTo(target_x, target_y);
    // Move to the target, only 30% of total distance to account for error
    driveFor(intialHeading, distance*0.3, 50);

    // Recalculate the heading and distance to the target
    double heading = calculateBearing(GPS.xPosition(distanceUnits::cm), GPS.yPosition(distanceUnits::cm), target_x, target_y);
    turnTo(heading, 15, 10);
    distance = distanceTo(target_x, target_y);
    // Move to the target, completing the remaining distance
    driveFor(heading, distance, 20);

    // Turn to the final target heading if specified, otherwise use current heading
    if (target_theta == -1){
        target_theta = GPS.heading();
    }
    turnTo(target_theta, 5, 2);
}

// Function to find the target object based on type and return its record
DETECTION_OBJECT findTarget(int type){
    DETECTION_OBJECT target;
    static AI_RECORD local_map;
    jetson_comms.get_data(&local_map);
    double lowestDist = 1000000;
    // Iterate through detected objects to find the closest target of the specified type
    for(int i = 0; i < local_map.detectionCount; i++) {
        double distance = distanceTo(local_map.detections[i].mapLocation.x, local_map.detections[i].mapLocation.y);
        if (distance < lowestDist && local_map.detections[i].classID == type) {
            target = local_map.detections[i];
            lowestDist = distance;
        }
    }
    return target;
}

// Function to drive to an object based on detection
void goToObject(OBJECT type){
    DETECTION_OBJECT target = findTarget(type);
    // If no target found, turn and try to find again
    if (target.mapLocation.x == 0 && target.mapLocation.y == 0){
        Drivetrain.turnFor(45, rotationUnits::deg, 50, velocityUnits::pct);
        target = findTarget(0);
    }
    // If the object is a ring, raise the arm so it can be picked up
    if (type == OBJECT::BlueRing || type == OBJECT::RedRing) {
        Arm.spinTo(300, rotationUnits::deg);
    }
    // Move to the detected target's position
    moveToPosition(target.mapLocation.x*100, target.mapLocation.y*100);
}

// Function to grab a ring when the arm is positioned over it
void grabRing() {
    Chain.spinFor(-3, rotationUnits::rev, false);
    Arm.spinTo(-45, rotationUnits::deg);
    Arm.spinTo(500, rotationUnits::deg, false);
    
}
// Function to drop the ring when the arm is postioned over a goal
void dropRing() {
    Arm.spinTo(500, rotationUnits::deg);
    Chain.spinFor(3, rotationUnits::rev, 40, velocityUnits::pct);
    Chain.spinFor(-3, rotationUnits::rev, false);
}
