#pragma once
#include "tjulib-chassis/basechassis.hpp"
#include <cmath>
extern double zero_drift_error;

namespace tjulib
{
    using namespace vex;

    class Oct_BaseChassis : public BaseChassis{
    private:
        T current_yaw = 0;
        T initial_yaw = 0;
        
    protected:
        std::vector<vex::motor*> &_lfMotors;
        std::vector<vex::motor*> &_lbMotors;
        std::vector<vex::motor*> &_rfMotors;
        std::vector<vex::motor*> &_rbMotors;
        const int deadzone = 15; 
        
    public: 
        
        Oct_BaseChassis(std::vector<std::vector<vex::motor*>*> &_chassisMotors, pidControl *_motorpidControl, Position*_position, const T _r_motor) : 
        BaseChassis(_chassisMotors, _motorpidControl, _position, _r_motor), _lfMotors(*chassisMotors[0]), _lbMotors(*chassisMotors[1]), _rfMotors(*chassisMotors[2]), _rbMotors(*chassisMotors[3]){}

        // 直接给电压的VRUN
        void VRUN(T lfspeed, T lbspeed, T rfspeed, T rbspeed)  {
            //printf("%d", _leftMotors.size());
            
            for (vex::motor *motor : (_lfMotors)){
                vexMotorVoltageSet(motor->index(), lfspeed * 12000.0 / 100.0);
            }
                
            for (vex::motor *motor : (_lbMotors)){
                vexMotorVoltageSet(motor->index(), lbspeed * 12000.0 / 100.0); 
            }
            for (vex::motor *motor : (_rfMotors)){
                
                vexMotorVoltageSet(motor->index(), rfspeed * 12000.0 / 100.0);
            }
                
            for (vex::motor *motor : (_rbMotors)){

                vexMotorVoltageSet(motor->index(), rbspeed * 12000.0 / 100.0); 
            }
        }
        
        // 打点移动
        void simpleMove(double vel, double sec)  {

        }

        // pid控制的轮子定速旋转 : 传入的速度参数为 inches/s ————注意，该函数仅且仅能够放到循环中进行连续PID控制，无法单独去使用
        void motorspin(motor *motor, T speed, T maxSpeed, T respondTime = 30, T gaptime = 5) override{
            timer time;
            time.clear();
            // 为实际转速，单位是 inches/s
            T target_vel = speed * r_motor;
            T current_vel = motor->velocity(dps) * r_motor;
            T error = (target_vel - current_vel);                                                                                           
            // 显然这只是一个P控制器
            motorpidControl -> resetpid();
            T voltage = motorpidControl->pidCalcu(error, maxSpeed);
            //printf("voltage : %lf, error : %lf\n", voltage, error);
            
            vexMotorVoltageSet(motor->index(), voltage * 12000.0 / 100.0);
            motorpidControl -> resetpid();

        }
        
        // pid控制的直线VRUN : 速度单位是inches/s
        void VRUNStable(T lfspeed, T lbspeed, T rfspeed, T rbspeed, T maxSpeed = 80)  {
           for (vex::motor *motor : (_lfMotors)){
                motorspin(motor, lfspeed, maxSpeed);
            }
                
            for (vex::motor *motor : (_lbMotors)){
                motorspin(motor, lbspeed, maxSpeed);
            }
            for (vex::motor *motor : (_rfMotors)){
                motorspin(motor, rfspeed, maxSpeed);
            }
                
            for (vex::motor *motor : (_rbMotors)){

                motorspin(motor, rbspeed, maxSpeed);
            }
        }

        // 设置停止类型 : brake coast hold
        void setBrakeType(vex::brakeType brake_type) override {btype = brake_type;}

        // 急刹 : 与setBrakeType一起配合使用
        void setStop(vex::brakeType type) override {
            for (vex::motor *motor : (_leftMotors))
                motor->stop(type);
            for (vex::motor *motor : (_rightMotors))
                motor->stop(type);
        }

        /*--------------  手动控制函数 ----------------*/
        // 无PID版本手动控制
        void ManualDrive_nonPID() override{

            // Retrieve the necessary joystick values
            T leftY = Controller1.Axis3.position(percent);
            T leftX = Controller1.Axis4.position(percent);
            T rightX = Controller1.Axis1.position(percent);
            if (abs(leftY) < deadzone) leftY = 0;            
            if (abs(leftX) < deadzone) leftX = 0;
            if (abs(rightX) < deadzone) rightX = 0;
            if(fabs(leftY) > 2 * fabs(leftX)){
                leftX = 0;
            }
            if(fabs(leftY) <= 2 * fabs(leftX)){
                leftY = 0;
            }
            VRUN(leftY + leftX + rightX, +leftY - leftX + rightX, -leftY + leftX + rightX, -leftY - leftX + rightX) ;


        }
        // PID版本手动控制
        void ManualDrive_PID() override{
            // 获取期望下车的最大速度
            T maxSpeed = 850 / sqrt(2);          // 最大速度开根号2，测量使用MaxSpeedTest
            // Retrieve the necessary joystick values
            T leftY = Controller1.Axis3.position(percent);
            T leftX = Controller1.Axis4.position(percent);
            T rightX = Controller1.Axis1.position(percent);
            rightX *= 0.7;
            if (abs(leftY) < deadzone) leftY = 0;            
            if (abs(leftX) < deadzone) leftX = 0;
            if (abs(rightX) < deadzone) rightX = 0;
            // 计算车X、Y方向的速度
            T v_LY = ( leftY / 100 ) * maxSpeed;
            T v_LX = ( leftX / 100 ) * maxSpeed;
            T v_RX = ( rightX / 100 ) * maxSpeed;

            if(fabs(leftY) > 2 * fabs(leftX)){
              //  v_LX = 0;
            }
            if(fabs(leftY) <= 2 * fabs(leftX)){
              //  v_LY = 0;
            }
            // 计算每个轮子的速度大小
            T v_lf = v_LY + v_LX + v_RX;
            T v_lb = v_LY - v_LX + v_RX;
            T v_rf = -v_LY + v_LX + v_RX;
            T v_rb = -v_LY - v_LX + v_RX;
            if(leftX == 0 && leftY == 0 && rightX == 0){   // 没有速度时候
                VRUN(0, 0, 0, 0);
            }else if(leftX != 0 || leftY != 0 || rightX != 0){    // 不旋转时
                VRUN(v_lf, v_lb, v_rf, v_rb);
            }else{
                VRUN(0, 0, 0, 0);
            }
            

        }
        
        /*--------------  电机调参测试函数 ----------------*/
        // 在自动中调用该函数可以获取八角底盘四个轮的最大转速
        void MaxSpeedTest(){
            VRUN(100, 100, 100, 100);
            while(1){
                T v_lf = 0;
                T v_lb = 0;
                T v_rf = 0;
                T v_rb = 0;
                for (vex::motor *motor : (_lfMotors)){
                    v_lf += motor->velocity(dps) * r_motor / _lfMotors.size();
                }   
                for (vex::motor *motor : (_lbMotors)){
                    v_lb += motor->velocity(dps) * r_motor / _lbMotors.size();
                }
                for (vex::motor *motor : (_rfMotors)){
                    v_rf += motor->velocity(dps) * r_motor / _rfMotors.size();
                }
                for (vex::motor *motor : (_rbMotors)){
                    v_rb += motor->velocity(dps) * r_motor / _rbMotors.size();
                }
                printf("v_lf : %lf, v_lb : %lf, v_rf : %lf, v_rb : %lf\n",v_lf,v_lb,v_rf,v_rb);
            }
            VRUN(0, 0, 0, 0);
        }

        // pid test : 用于调PID参数
        void MotorPIDTest(T lfspeed, T lbspeed, T rfspeed, T rbspeed, T testtime = 3000, T maxSpeed = 80){
            
            timer time;
            time.clear();
            while(1){
                if(time.time(msec) >= testtime){
                    break;
                }
                T v_lf = 0;
                T v_lb = 0;
                T v_rf = 0;
                T v_rb = 0;
                for (vex::motor *motor : (_rfMotors)){
                    motorspin(motor, rfspeed, maxSpeed);
                    v_rf += motor->velocity(dps) * r_motor / _rfMotors.size();
                }
                for (vex::motor *motor : (_lfMotors)){
                    motorspin(motor, lfspeed, maxSpeed);
                    v_lf += motor->velocity(dps) * r_motor / _lfMotors.size();
                }   
                for (vex::motor *motor : (_lbMotors)){
                    motorspin(motor, lbspeed, maxSpeed);
                    v_lb += motor->velocity(dps) * r_motor / _lbMotors.size();
                }
                
                for (vex::motor *motor : (_rbMotors)){
                    motorspin(motor, rbspeed, maxSpeed);
                    v_rb += motor->velocity(dps) * r_motor / _rbMotors.size();
                }
                printf("v_lf : %lf, v_lb : %lf, v_rf : %lf, v_rb : %lf\n",v_lf,v_lb,v_rf,v_rb);
            }
            VRUN(0, 0, 0, 0);
        }

    };

} // namespace tjulib
