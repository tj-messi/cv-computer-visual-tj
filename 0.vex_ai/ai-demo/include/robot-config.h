using namespace vex;

/*************************************

        physical configurations

*************************************/
// �?�?
extern const double car_width;
// 车轮半径
extern const double r_wheel;
// 底盘电机-�?的齿�?传动�?
extern const double gear_ratio;
// 电机�?�?-电机�?周的换算�?
extern const double r_motor; 
// 一�?地垫长度(inches)
extern const double cell;
// 里程计偏�?（inches�?----做垂�?
extern const double hOffset;
extern const double vOffset;
// 编码�?周长
extern const double r_wheel_encoder;
// GPS的x轴方向偏�? 
extern const double gps_offset_x;      
// GPS的y轴方向偏�?                
extern const double gps_offset_y;
// 编码�?旋转角度
extern const double encoder_rotate_degree;
extern double gps_x;
extern double gps_y;
extern double gps_heading;    
//vision控制
extern bool photoFlag;
extern bool abandon;
extern bool throwFlag;
extern bool reverseSpin;
extern bool forwardSpin;      
extern bool ring_convey_spin;      // �?否开始进行吸�?
extern int ring_color;             // 对获取的�?的�?�色进�?��?�查，0�?没有�?�?1�?蓝色�?�?2�?红色�?           
/*************************************

            VEX devices

*************************************/
extern brain Brain;
extern motor L1;
extern motor L2;
extern motor L3;
extern motor L4;
extern motor R1;
extern motor R2;
extern motor R3;
extern motor R4;

extern motor_group lift_arm;
extern motor_group roller_group;

extern controller Controller1;
extern motor side_bar;

extern vex::message_link AllianceLink;

extern encoder encoderHorizonal;
extern encoder encoderVertical;

extern pwm_out gas_push;
extern pwm_out gas_lift;
extern pwm_out gas_hold;

extern inertial imu;

extern distance DistanceSensor;

extern vex::gps GPS_;

extern vision Vision;

extern motor convey_beltMotorA;
extern motor convey_beltMotorB;
extern motor_group convey_belt;

extern std::vector<vision::signature*>Red;
extern std::vector<vision::signature*>Blue;

// 四�?�底盘电机组声明
extern std::vector<vex::motor*> _leftMotors;
extern std::vector<vex::motor*> _rightMotors;

// �?角底盘电机组声明
extern std::vector<vex::motor*> _lfMotors;
extern std::vector<vex::motor*> _lbMotors;
extern std::vector<vex::motor*> _rfMotors;
extern std::vector<vex::motor*> _rbMotors;
// vision signature
extern vision::signature Red1;
extern vision::signature Red2;
extern vision::signature Red3;
extern vision::signature Blue1;
extern vision::signature Blue2;
extern vision::signature Blue3;


extern std::vector<vision::signature*>Red;
extern std::vector<vision::signature*>Blue;
// extern pwm_out gas;

/**
 * Used to initialize code/tasks/devices added using tools in VEXcode Pro.
 * 
 * This should be called at the start of your int main function.
 */
void  vexcodeInit( void );