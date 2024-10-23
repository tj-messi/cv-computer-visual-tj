#include "vex.h"
#include<vector>
using namespace vex;

/*************************************

        physical configurations

*************************************/
const double car_width = 11.15;                   // č˝?čˇ?
const double r_wheel = 4 / 2;                     // č˝Śč˝Žĺĺž
const double gear_ratio = 0.667;                  // ĺşççľćş-č˝?çé˝żč˝?äź ĺ¨ćŻďźĺ ééćŻĺ°ąĺ¤§äş1ďźĺééćŻĺ°ąĺ°äş1ďź?
const double r_motor = r_wheel * gear_ratio ;     // çľćşč˝?č§?-çľćşč˝?ĺ¨çć˘çŽćŻ?
const double cell  = 24;                          // ä¸ä¸?ĺ°ĺŤéżĺşŚ(inches)
const double hOffset  = -5;                       // éç¨čŽĄĺç˝?ďźinchesďź?----äťćč˝?ä¸?ĺżĺéç¨čŽĄč˝Žĺťśäź¸ćšĺä˝ĺçş?
const double vOffset  = 5;                        // éç¨čŽĄĺç˝?ďźinchesďź?----äťćč˝?ä¸?ĺżĺéç¨čŽĄč˝Žĺťśäź¸ćšĺä˝ĺçş?
const double r_wheel_encoder = 2.75 / 2;          // çźç č˝?ĺ¨éż
const double gps_offset_x = 0;                    // GPSçxč˝´ćšĺĺç˝? 
const double gps_offset_y = 6.7;                  // GPSçyč˝´ćšĺĺç˝? 
const double encoder_rotate_degree = 45;          // çźç č˝?ćč˝Źč§ĺşŚ
/*************************************

            state flags

*************************************/
// gpsĺć (ĺ¸Śćč˝?ä¸?ĺżĺç˝?äż?ć­?)
double gps_x = 0;
double gps_y = 0;
double gps_heading = 0;
//visionć§ĺś
bool photoFlag = false;
bool abandon = true;
bool throwFlag = false;
bool reverseSpin = false;
bool forwardSpin = false;

bool ring_convey_spin = false;  // ć?ĺŚĺźĺ§čżčĄĺ¸ç?
int ring_color = 0;             // ĺŻščˇĺçç?çé?č˛čżč?ć?ćĽďź0ć?ć˛Ąćç?ďź?1ć?čč˛ç?ďź?2ć?çş˘č˛ç?
 
/*************************************

            VEX devices

*************************************/
// A global instance of brain used for printing to the V5 Brain screen
brain  Brain;
// ĺşççľćş - ĺč?ĺşç?
motor L1 = motor(PORT13, ratio6_1, false);
motor L2 = motor(PORT13, ratio6_1, true);
motor L3 = motor(PORT13, ratio6_1, false);
motor L4 = motor(PORT13, ratio6_1, true);
motor R1 = motor(PORT13, ratio6_1, true);
motor R2 = motor(PORT13, ratio6_1, false);
motor R3 = motor(PORT13, ratio6_1, true);
motor R4 = motor(PORT13, ratio6_1, false); 
std::vector<vex::motor*> _leftMotors = {&L1, &L2, &L3,  &L4};
std::vector<vex::motor*> _rightMotors = {&R1, &R2, &R3, &R4};

// ĺşççľćş - ĺ?č§ĺşç?
motor lf1 = motor(PORT1, ratio18_1, false);
motor lf2 = motor(PORT2, ratio18_1, true);
motor lb1 = motor(PORT3, ratio18_1, false);
motor lb2 = motor(PORT4, ratio18_1, true);
motor rf1 = motor(PORT10, ratio18_1, false);
motor rf2 = motor(PORT9, ratio18_1, true);
motor rb1 = motor(PORT11, ratio18_1, false);
motor rb2 = motor(PORT12, ratio18_1, true);

std::vector<vex::motor*> _lfMotors = {&lf1, &lf2};
std::vector<vex::motor*> _lbMotors = {&lb1, &lb2};
std::vector<vex::motor*> _rfMotors = {&rf1, &rf2};
std::vector<vex::motor*> _rbMotors = {&rb1, &rb2};

// ć?ĺč
motor lift_armMotorA = motor(PORT8, ratio36_1, true);
motor lift_armMotorB = motor(PORT15, ratio36_1, false);
motor_group lift_arm = motor_group(lift_armMotorA, lift_armMotorB);

// äź éĺ¸Ś
motor convey_beltMotorA = motor(PORT18, ratio36_1, true);
motor convey_beltMotorB = motor(PORT14, ratio36_1, false);
motor_group convey_belt = motor_group(convey_beltMotorA, convey_beltMotorB);

// ĺ¸ç
motor rollerMotorA = motor(PORT6, ratio18_1, true);
motor rollerMotorB = motor(PORT13, ratio18_1, false);
motor_group roller_group = motor_group(rollerMotorA, rollerMotorB);
// éĽć§ĺ?
controller Controller1 = controller(primary);
// éäżĄĺ¤Šçşż
vex::message_link AllianceLink(PORT13, "tju1", linkType::worker);
// éç¨čŽ?
encoder encoderHorizonal = encoder(Brain.ThreeWirePort.A);
encoder encoderVertical = encoder(Brain.ThreeWirePort.G);
// ĺŻźĺĽć?
motor side_bar = motor(PORT13, ratio18_1, false);
// imuć?ć§äź ćĺ¨
inertial imu = inertial(PORT17);  // çŹ?äşä¸Şĺć°čŚĺright

// ć°ĺ¨äť?
pwm_out gas_push = pwm_out(Brain.ThreeWirePort.D);
pwm_out gas_lift = pwm_out(Brain.ThreeWirePort.E);
pwm_out gas_hold = pwm_out(Brain.ThreeWirePort.F);

// čˇç?ťäź ćĺ¨
distance DistanceSensor = distance(PORT13);
// gps
gps GPS_ = gps(PORT16, 0, 0, inches, 0);

// vision signature
vision::signature Red1 = vision::signature(3, 9051, 11375, 10213, -1977, -833, -1405, 1.6711680, 0);
vision::signature Red2 = vision::signature(3,  5461, 8761, 7111, -1457, -167, -812, 0.8144962, 0);
vision::signature Red3 = vision::signature(3,  8191, 9637, 8914, -1831, -735, -1283, 0.6828844, 0);

vision::signature Blue1 = vision::signature(2,  -4177, -3545, -3861, 6099, 7047, 6573, 1.2, 0);
vision::signature Blue2 = vision::signature(2, -5461, -3919,  -4690, 7721, 11045, 9383, 1.123142, 0);
vision::signature Blue3 = vision::signature(1,  -4177, -3545, -3861, 6099, 7047, 6573, 1.2, 0);
std::vector<vision::signature*>Red = {&Red1, &Red2, &Red3};
std::vector<vision::signature*>Blue = {&Blue1, &Blue2, &Blue3};
// vision
vision Vision = vision(PORT19, 50, Blue1, Blue2, Blue3);


// VEXcode generated functions
// define variable for remote controller enable/disable
bool RemoteControlCodeEnabled = true;

// manager and worker robots
// Comment out the following definition to build for the worker robot
#define  MANAGER_ROBOT    1

#if defined(MANAGER_ROBOT)
#pragma message("building for the manager")
ai::robot_link       link( PORT13, "robot_32456_1", linkType::manager );
#else
#pragma message("building for the worker")
ai::robot_link       link( PORT13, "robot_32456_1", linkType::worker );
#endif

/*************************************

            axis defination

*************************************/
/*

robot_local : 

                  ^ y   Head  ^    ^--->
                  |           |    |<-- imu  
                  |
                  |
(rotation_center) |ââââââââââ?> x


robot_global : 

                  ^ y
                  |           
                  |
                  |
         (middle) |ââââââââââ?> x

*/

/**
 * Used to initialize code/tasks/devices added using tools in VEXcode Pro.
 * 
 * This should be called at the start of your int main function.
 */
void vexcodeInit( void ) {
  // nothing to initialize
}