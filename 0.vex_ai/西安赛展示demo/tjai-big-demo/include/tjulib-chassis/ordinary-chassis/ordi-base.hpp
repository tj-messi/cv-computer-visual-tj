#pragma once
#include "tjulib-chassis/basechassis.hpp"

extern double zero_drift_error;

namespace tjulib
{
    using namespace vex;

    class Ordi_BaseChassis : public BaseChassis{
    private:
        T current_yaw = 0;
        T initial_yaw = 0;
        
    protected:
        std::vector<vex::motor*> &_leftMotors ;
        std::vector<vex::motor*> &_rightMotors;
        const int deadzone = 5; 
        
    public: 
        
        Ordi_BaseChassis(std::vector<std::vector<vex::motor*>*> &_chassisMotors, pidControl *_motorpidControl, Position*_position, const T _r_motor) : 
        BaseChassis(_chassisMotors, _motorpidControl, _position, _r_motor), _leftMotors(*chassisMotors[0]), _rightMotors(*chassisMotors[1]){}

        // 直接给电压的VRUN
        void VRUN(T Lspeed, T Rspeed) {
            //printf("%d", _leftMotors.size());
            
            for (vex::motor *motor : (_leftMotors)){
                
                vexMotorVoltageSet(motor->index(), Lspeed * 12000.0 / 100.0);
            }
                
            for (vex::motor *motor : (_rightMotors)){
                //
                vexMotorVoltageSet(motor->index(), Rspeed * 12000.0 / 100.0); 
            }
        }
        
        // 打点移动
        void simpleMove(double vel, double sec) {
            VRUN(vel, vel);
            task::sleep(1000*sec);
            VRUN(0, 0);
            task::sleep(20);
        }

        // pid控制的轮子定速旋转 : 传入的速度参数为 inches/s
        void motorspin(motor *motor, T speed, T maxSpeed, T respondTime = 100, T gaptime = 20) override {
            timer time;
            time.clear();
            // 为实际转速，单位是 inches/s
            T target_vel = speed * r_motor;
            T current_vel = motor->velocity(dps);
            T error = (target_vel - current_vel);                                                                                             motorpidControl -> resetpid();

            // 直接计算pid目标
            while(motorpidControl -> overflag() || time.time(msec) >= respondTime){
                T voltage = motorpidControl->pidCalcu(error, 100);
                vexMotorVoltageSet(motor->index(), voltage * 12000.0 / 100.0);
                // 控制周期
                task::sleep(gaptime);
            }
            motorpidControl -> resetpid();
        }
        
        // pid控制的直线VRUN : 速度单位是inches/s
        void VRUNStraight(T Lspeed, T Rspeed, T max_delta_voltage) {
            for (vex::motor *motor : (_leftMotors)){
                motorspin(motor, Lspeed, 80);
            }
                
            for (vex::motor *motor : (_rightMotors)){
                motorspin(motor, Rspeed, 80);
            }
        }
        // // pid控制的直线VRUN
        // void VRUNStraight(T Lspeed, T Rspeed, T max_delta_voltage){
        //    // current_yaw = Math::getWrap360(imu.rotation());
        //     //angleWrap(initial_yaw, current_yaw);
        //     // 获取两侧轮初始转速
        //     T leftMotorsvel = 0, rightMotorsvel = 0;
        //     for (vex::motor *motor : (_leftMotors)){
        //         leftMotorsvel += motor->velocity(dps) / _leftMotors.size();
        //     }

        //     for (vex::motor *motor : (_rightMotors)){
        //         rightMotorsvel += motor->velocity(dps) / _rightMotors.size();
        //     }
        //     T target_vel = (leftMotorsvel + rightMotorsvel) / 2;
        //     T delta_left_voltage = 0, delta_right_voltage = 0;
        //     // 计算速度
        //     T left_error = (target_vel - leftMotorsvel);
        //     T right_error = (target_vel - rightMotorsvel);

        //    // printf("left: %lf \n", leftMotorsvel);
        //    // printf("right: %lf \n", rightMotorsvel);
            
        //   //  if(std::fabs(current_yaw - initial_yaw) > 2){
        //    // if(std::fabs(target_vel) > 2300){
        //        // printf("imu_current: %lf\n", current_yaw);
        //         //printf("imu_initial: %lf\n", initial_yaw);
        //         left_error *=  0.075;
        //         right_error *=  0.075;
        //      //   printf("left_error: %lf \n", left_error);
        //      //   printf("right_error: %lf \n", right_error);
        //         // 通过速度差计算电压修正补偿（增量pid控制）
        //         delta_left_voltage = StraightLineControl->pidCalcu(left_error, max_delta_voltage);
        //         delta_right_voltage = StraightLineControl->pidCalcu(right_error, max_delta_voltage);
        //         printf("delta_left_voltage: %lf \n", delta_left_voltage);
        //         printf("delta_right_voltage: %lf \n", delta_right_voltage);
        //     //}
                
        //     //printf("delta_left_voltage: %lf \n", delta_left_voltage);
        //    // printf("delta_right_voltage: %lf \n", delta_right_voltage);
        //     VRUN(Lspeed + delta_left_voltage, Rspeed + delta_right_voltage);
            
        //    /*
        //     current_yaw = Math::getWrap360(imu.rotation());
        //     angleWrap(initial_yaw, current_yaw);
        //     T delta_voltage = 0;
        //     printf("delta_left_voltage: %lf \n", std::fabs(current_yaw - initial_yaw));
        //     if(std::fabs(current_yaw - initial_yaw) > 3){
                
        //         // 通过速度差计算电压修正补偿（增量pid控制）
        //         delta_voltage = StraightLineControl->pidCalcu(initial_yaw, max_delta_voltage, current_yaw);
        //     }
        
        //     printf("delta_left_voltage: %lf \n", delta_voltage);
        //     VRUN(Lspeed - delta_voltage, Rspeed + delta_voltage);
        //       */
        // }


        // 设置停止类型 : brake coast hold
        void setBrakeType(vex::brakeType brake_type) override {btype = brake_type;}

        // 急刹 : 与setBrakeType一起配合使用
        void setStop(vex::brakeType type) override {
            for (vex::motor *motor : (_leftMotors))
                motor->stop(type);
            for (vex::motor *motor : (_rightMotors))
                motor->stop(type);
        }

        // 无PID版本手动控制
        void ManualDrive_nonPID() override{

            for (vex::motor *motor : (_leftMotors)){
               // printf("LeftMotor_vel : %lf \n", motor->velocity(dps));
            }
                
            for (vex::motor *motor : (_rightMotors)){
                //printf("RightMotor_vel : %lf \n", motor->velocity(dps));
            }
            // Retrieve the necessary joystick values
            int leftY = Controller1.Axis3.position(percent);
            int rightX = Controller1.Axis1.position(percent);
            if (abs(leftY) < deadzone) leftY = 0;            
            if (abs(rightX) < deadzone) rightX = 0;

            //rightX = rightX * 0.7;
            VRUN(leftY+rightX, leftY-rightX);


        }
        // PID版本手动控制
        void ManualDrive_PID() override{
            for (vex::motor *motor : (_leftMotors)){
               // printf("LeftMotor_vel : %lf \n", motor->velocity(dps));
            }
                
            for (vex::motor *motor : (_rightMotors)){
                //printf("RightMotor_vel : %lf \n", motor->velocity(dps));
            }
            // Retrieve the necessary joystick values
            int leftY = Controller1.Axis3.position(percent);
            int rightX = Controller1.Axis1.position(percent);
            if (abs(leftY) < deadzone) leftY = 0;            
            if (abs(rightX) < deadzone) rightX = 0;

            if(leftY!=0 && rightX==0){
                VRUNStraight(leftY+rightX, leftY-rightX, leftY * 0.55);
            }else if(rightX!=0){
                VRUN(leftY+rightX, leftY-rightX);
                initial_yaw = Math::getWrap360(imu.rotation());
            }else{
                VRUN(0, 0);

                initial_yaw = Math::getWrap360(imu.rotation());
            }

            //rightX = rightX * 0.7;
        }

    };

} // namespace tjulib
