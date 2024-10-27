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
#include "vex-ai/ai_functions.h"
#include <string>
#include <iostream>
using namespace vex;
using namespace std;

// Calculates the distance to the coordinates from the current robot position
double distanceTo(double target_x, double target_y){
    double distance = sqrt(pow((target_x - gps_x), 2) + pow((target_y - gps_y), 2));
    return distance;
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

