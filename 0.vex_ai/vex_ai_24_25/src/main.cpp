/*----------------------------------------------------------------------------*/
/*                                                                            */
/*    Module:       main.cpp                                                  */
/*    Author:       james                                                     */
/*    Created:      Mon Aug 31 2020                                           */
/*    Description:  V5 project                                                */
/*                                                                            */
/*----------------------------------------------------------------------------*/

// ---- START VEXCODE CONFIGURED DEVICES ----
// ---- END VEXCODE CONFIGURED DEVICES ----
#include "ai_functions.h"

using namespace vex;
controller Controller;
brain Brain;
// Robot configuration code.
motor leftDrive = motor(PORT1, ratio18_1, false);
motor rightDrive = motor(PORT2, ratio18_1, true);
gps GPS = gps(PORT12, -127, -165, distanceUnits::mm, 180);
smartdrive Drivetrain = smartdrive(leftDrive, rightDrive, GPS, 319.19, 320, 40, mm, 1);
// Controls arm used for raising and lowering rings
motor Arm = motor(PORT2, ratio18_1, false);
// Controls the chain at the front of the arm
// used for pushing rings off of the arm
motor Chain = motor(PORT8, ratio18_1, false);

pwm_out lift = pwm_out
// A global instance of competition
competition Competition;

// create instance of jetson class to receive location and other
// data from the Jetson nano
//
ai::jetson  jetson_comms;

/*----------------------------------------------------------------------------*/
// Create a robot_link on PORT1 using the unique name robot_32456_1
// The unique name should probably incorporate the team number
// and be at least 12 characters so as to generate a good hash
//
// The Demo is symetrical, we send the same data and display the same status on both
// manager and worker robots
// Comment out the following definition to build for the worker robot
#define  MANAGER_ROBOT    1

#if defined(MANAGER_ROBOT)
#pragma message("building for the manager")
ai::robot_link       link( PORT10, "robot_32456_1", linkType::manager );
#else
#pragma message("building for the worker")
ai::robot_link       link( PORT10, "robot_32456_1", linkType::worker );
#endif

void auto_Isolation(void) {
  while(1)
  {
     // Calibrate GPS Sensor
  GPS.calibrate();
  // Optional wait to allow for calibration
  waitUntil(!(GPS.isCalibrating()));

  // Set brake mode for the arm
  Arm.setStopping(brakeType::hold);
  // Reset the position of the arm while its still on the ground
  Arm.resetPosition();
  // Lift the arm to prevent dragging
  Arm.spinTo(75, rotationUnits::deg);

  // Finds and moves robot to over the closest blue ring
  goToObject(OBJECT::BlueRing);
  grabRing();
  // Find and moves robot to the closest mobile drop
  // then drops the ring on the goal
  goToObject(OBJECT::MobileGoal);
  dropRing();
  // Back off from the goal
  Drivetrain.driveFor(-30, distanceUnits::cm);
  }
}


void auto_Interaction(void) {
  while(1)
  {
    Arm.setVelocity(60, percent);
  }
}

bool firstAutoFlag = true;

void autonomousMain(void) {

  if(firstAutoFlag)
    auto_Isolation();
  else 
    auto_Interaction();

  firstAutoFlag = false;
}

void VRUN(double l,double r)
{
  vexMotorVoltageSet(vex::PORT15,-l*120); 
  vexMotorVoltageSet(vex::PORT5,l*120); 
  vexMotorVoltageSet(vex::PORT1,-l*120); 
  //vexMotorVoltageSet(vex::PORT,-l*120);
  vexMotorVoltageSet(vex::PORT11,r*120);
  vexMotorVoltageSet(vex::PORT13,-r*120);
  vexMotorVoltageSet(vex::PORT4,r*120);
  //vexMotorVoltageSet(vex::PORT4,-r*120);
}

int main()
{
  while(1)
{
/***操纵***/
int fb,lf;
/********************************************
相应Axis对应(两个十字对应手柄左右两边遥感，可能有误)：
Axis1
=
Axis2 =====
= Axis3
=
Axis4===
=
*********************************************/
fb=Controller.Axis3.value();
lf=Controller.Axis4.value();
fb=std::abs(fb)>15?fb:0;
lf=std::abs(lf)>15?lf:0;
if(fb!=0||lf!=0) VRUN((fb+lf)*100.0/127.0,(fb-lf)*100.0/127.0);
else VRUN(0,0);
//提前定义好了投盘电机ShootMotor
//按Y键转动，松开停止

Controller.ButtonL1.pressed([]() {        
        Arm.spin(fwd);
      Chain.spin(fwd);// 电机正转
    });
Controller.ButtonL1.released([]() {        
        Arm.stop(hold);
        Chain.stop(hold);
    });
    
Controller.ButtonR1.pressed([]() {        
       lift.state(100,pct);
    });
Controller.ButtonR1.released([]() {        
       lift.state(0,pct);
    });

}
}