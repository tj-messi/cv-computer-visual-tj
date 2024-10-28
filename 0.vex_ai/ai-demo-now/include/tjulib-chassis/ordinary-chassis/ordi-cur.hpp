#pragma once
#include "tjulib-chassis/ordinary-chassis/ordi-straight.hpp"

extern double zero_drift_error;
namespace tjulib
{
    using namespace vex;
    class Ordi_CurChassis : virtual public Ordi_BaseChassis{
    protected:
        T b; // car width
        pidControl *circleControl = NULL;
    protected:
      
    public:
        Ordi_CurChassis(std::vector<std::vector<vex::motor*>*> &_chassisMotors, pidControl *_motorpidControl, Position*_position, const T _r_motor, pidControl *cur_pid, T car_width)
            : Ordi_BaseChassis(_chassisMotors, _motorpidControl, _position, _r_motor), circleControl(cur_pid), b(car_width){}

        // 计算转向算子,事实上这个算子应该是一个接近固定的值,因为他可以用一个定圆的半径+车宽俩常量控制，而且通过曲线控制可以保证车沿着定圆运动
        T GetCircleControlGamma(T targetDeg, T currentAngle, T distance){
            // 控制差角
            T delta_theta = targetDeg - currentAngle;
            printf("delta_theta: %lf\n",delta_theta);
            printf("t1:%lf\n",(double)(b/distance));
            printf("sin:%lf\n",sin(delta_theta*0.5));
            // 计算算子，最后一小段因为很小，直接计算会出现一些问题，因此考虑直接返回0了
            if(std::fabs(delta_theta)<2){
                return 0;
            }else{
                return (double)(b/distance)*sin(0.5*delta_theta);
            }
        
        }
        // 圆弧控制的设计需求是溜边走
        //圆弧移动控制：参数：目标角度target_angle，距离distance,车横向宽度car_width，最大控制速度maxSpeed，最大控制时间maxtime_ms，方向fwd（正方向1是向前进）
        // 有个小bug还没有修复，就是现在只能走小圆弧不能走大圆，走的方向一定要和转角匹配
        void CircleTurnMove(T target_angle, T distance, T maxSpeed, double maxtime_ms, int fwd = 1){
            timer mytime;
            mytime.clear();
            double totaltime = 0;
            T finalTurnSpeed = 20;

            double targetDeg = Math::getWrap360(target_angle); // Obtain the closest angle to the target position
            double currentAngle = Math::getWrap360(imu.rotation());

            double prev_speed = finalTurnSpeed;

            int init =0;
            
            T error = optimalTurnAngle(targetDeg, currentAngle);
            
            // 零误差漂移修正0
            currentAngle = imu.angle() - zero_drift_error;

            int gamma = 0; // 转向算子
            // 计算转向算子,转向算子是一个恒定值，因此只要一开始计算一次即可
            gamma = GetCircleControlGamma(targetDeg, currentAngle, distance);

            circleControl->resetpid();

            //  圆弧转向控制
            while (!circleControl->overflag() || (fabs(error) >= 2)) // If within acceptable distance, PID output is zero.
            {
                //printf("gamma: %lf",(double)(car_width/distance)*sin(3.14159*0.5*(targetDeg-currentAngle)/180));
                // printf("targetDeg: %f \n", targetDeg);
                // printf("currentAngle: %f \n", currentAngle);
                // printf("speed: %f \n", finalTurnSpeed);
                // printf("cnt: %d \n", cnt);
                // printf("error: %f \n", std::fabs(targetDeg - currentAngle));
                // 最大时间限制
                if(totaltime=mytime.time(msec)>=maxtime_ms){
                    break;
                }
                // 震荡回调控制
                if(std::fabs(error) < circleControl->params->errorThreshold && finalTurnSpeed <= circleControl->params->minSpeed){
                    circleControl->cnt++;
                }
                
                // 零误差漂移修正
                currentAngle = imu.angle() - zero_drift_error;
                
                // 角度修正，通过角度修正，一方面把target和current两个angle都变换到-360~360之间，另一方面也变换为控制目标差值直接target-current的情况
                error = optimalTurnAngle(targetDeg, currentAngle);
                
                // Plug angle into turning PID and get the resultant speed
                finalTurnSpeed = circleControl->pidCalcu(error, maxSpeed); 
                // 反向调整限速(我们不希望回调的时候会回太快，要不然震荡会很猛烈)
                if(finalTurnSpeed*prev_speed<0&& init > 0){
                    maxSpeed *= 0.2;
                }
               // printf("gamma:%lf tartgetDeg:%lf currentAngle:%lf\n", gamma,targetDeg,currentAngle);
                init = 1;
                // 更新速度
                prev_speed = finalTurnSpeed;
                if(!fwd){ finalTurnSpeed = -finalTurnSpeed; }
                VRUN((1-gamma)*finalTurnSpeed, (1+gamma)*finalTurnSpeed);
                task::sleep(5);
            }

            circleControl->resetpid();

            VRUN(0, 0);
            setStop(vex::brakeType::brake);
        }
        
        void CurPIDMove(Point target, T maxSpeed, T maxtime_ms, int fwd = 1){
            T distance = GetDistance(target);
            T target_angle = target.angle;
            CircleTurnMove(target_angle, distance, maxSpeed, maxtime_ms, fwd);

        }
    };
};