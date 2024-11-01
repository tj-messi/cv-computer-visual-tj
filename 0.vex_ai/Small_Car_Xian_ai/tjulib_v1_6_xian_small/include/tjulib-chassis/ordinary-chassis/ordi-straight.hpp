#pragma once
#include "tjulib-chassis/ordinary-chassis/ordi-base.hpp"

extern double zero_drift_error;

namespace tjulib
{
    using namespace vex;

    class Ordi_StraChassis : virtual public Ordi_BaseChassis{
    protected:
        pidControl *fwdControl = NULL;      // 直线移动pid控制器   
        pidControl *turnControl = NULL;     // 转向pid控制器
    private:
        const double PI = 3.14159265358979323846;

    protected:
    
    public:
        Ordi_StraChassis(std::vector<std::vector<vex::motor*>*> &_chassisMotors, pidControl *_motorpidControl, Position*_position, const T _r_motor, pidControl *_fwdpid, pidControl *_turnpid):
            Ordi_BaseChassis(_chassisMotors, _motorpidControl, _position, _r_motor), fwdControl(_fwdpid), turnControl(_turnpid) {}
        
        // 转向角pid
        void turnToAngle(double angle, T maxSpeed, double maxtime_ms, int fwd = 1){
            timer mytime;
            mytime.clear();
            double totaltime = 0;
            T finalTurnSpeed = 20;
            
            double targetDeg = Math::getWrap360(angle); // Obtain the closest angle to the target position
            double currentAngle = Math::getWrap360(imu.rotation());

            double prev_speed = finalTurnSpeed;

            T error = optimalTurnAngle(targetDeg, currentAngle);
            printf("targetDeg : %lf, currentAngle : %lf", targetDeg, currentAngle);
            int init =0;

            turnControl->resetpid();

            while (!turnControl->overflag() || (fabs(error) >= 2)) // If within acceptable distance, PID output is zero.
            {
                
                if(totaltime=mytime.time(msec)>=maxtime_ms){
                    break;
                }

                if(std::fabs(error) < turnControl->params->errorThreshold && finalTurnSpeed <= turnControl->params->minSpeed){
                    turnControl->cnt++;
                }

                currentAngle = imu.angle() - zero_drift_error;

                if(fwd)
                    error = optimalTurnAngle(targetDeg, currentAngle);
                else
                    error = targetDeg - currentAngle;
               // printf("targetDeg : %lf, currentAngle : %lf error : %lf \n", targetDeg, currentAngle, error);
                finalTurnSpeed = turnControl->pidCalcu(error, maxSpeed); // Plug angle into turning PID and get the resultant speed
        
                if(finalTurnSpeed*prev_speed<0&& init > 0){
                    maxSpeed *= 0.2;
                }
                init = 1;
     
                prev_speed = finalTurnSpeed;
                
                VRUN(finalTurnSpeed, -finalTurnSpeed);
                task::sleep(5);
            }

            turnControl->resetpid();

            VRUN(0, 0);
            setStop(vex::brakeType::brake);
        }

        // distance of base (inches)
        void moveInches(T inches, T maxSpeed, double maxtime_ms = 5000, int fwd = 1){
            timer mytime;
            mytime.clear();
            T finalFwdSpeed = 20;
            T targetDistant = inches;
            T startError = (position->LeftBaseDistance + position->RightBaseDistance) / 2;
            fwdControl->resetpid();
           
            while (!fwdControl->overflag()) // If within acceptable distance, PID output is zero.
            {
                targetDistant = inches - fabs(startError - (position->LeftBaseDistance + position->RightBaseDistance)/2) ; // Obtain the closest angle to the target position
                //printf("error1 %lf \n", targetDistant);
                if(std::fabs(targetDistant)<=fwdControl->params->errorThreshold ){
                    fwdControl->cnt++;
                }

                // printf("cnt %lf \n", fwdControl->cnt);
                // printf("speed %lf \n", finalFwdSpeed);
                
                
                if(mytime.time(msec)>=maxtime_ms){
                    break;
                }
                finalFwdSpeed = fwdControl->pidCalcu(targetDistant, maxSpeed); // Plug angle into turning PID and get the resultant speed 
                
                if(!finalFwdSpeed) finalFwdSpeed = 2;
                if(!fwd) finalFwdSpeed = -finalFwdSpeed;
                VRUN(finalFwdSpeed, finalFwdSpeed);
               // VRUN(finalFwdSpeed, finalFwdSpeed);

                task::sleep(25);

            }
            VRUN(0, 0);
            fwdControl->resetpid();
        }
        
        // 基于距离传感器读数的pid
        void DistanceSensorMove(T mms, T maxSpeed, double maxtime_ms = 5000, int fwd = 1){
            timer mytime;
            mytime.clear();
            T finalFwdSpeed = 20;
            T targetDistant = mms;
            T startError = DistanceSensor.objectDistance(mm);
            fwdControl->resetpid();
           
            while (!fwdControl->overflag()) // If within acceptable distance, PID output is zero.
            {
                if(targetDistant<=5 && finalFwdSpeed <= 15){
                    fwdControl->cnt++;
                }
                  printf("error: %lf distance: %lf \n", targetDistant, DistanceSensor.objectDistance(mm));
                // printf("cnt %lf \n", fwdControl->cnt);
                // printf("speed %lf \n", finalFwdSpeed);
                targetDistant = -(mms - fabs(DistanceSensor.objectDistance(mm)) ); // Obtain the closest angle to the target position
                
                if(mytime.time(msec)>=maxtime_ms){
                    break;
                }
                finalFwdSpeed = fwdControl->pidCalcu(targetDistant, maxSpeed); // Plug angle into turning PID and get the resultant speed 
                
                if(!finalFwdSpeed) finalFwdSpeed = 2;
                if(!fwd) finalFwdSpeed = -finalFwdSpeed;
                VRUN(finalFwdSpeed, finalFwdSpeed);

                task::sleep(30);
            }
            VRUN(0, 0);
            fwdControl->resetpid();
        }

        void turnToTarget(Point target, T maxSpeed, double maxtime_ms, int fwd){
            // we set robot angle as the angle between heading and the x axis
            // the function getDegree is used to calculate angle for x axis
            turnToAngle(getDegree(target), maxSpeed, maxtime_ms, fwd);
        }
       
        void moveToTarget(Point target, T maxFwdSpeed, T maxTurnSpeed, double maxtime_ms, int fwd)
        {
            timer mytime;
            mytime.clear();
            double totaltime = 0;
            
            turnToTarget(target, maxTurnSpeed, maxtime_ms / 2, fwd);

            T finalFwdSpeed = 3;

            T targetDistant;
            while (fabs(finalFwdSpeed) > 2 &&(totaltime=mytime.time(msec)<maxtime_ms)) // If within acceptable distance, PID output is zero.
            {
                targetDistant = GetDistance(target); // Obtain the closest angle to the target position
                finalFwdSpeed = fwdControl->pidCalcu(targetDistant, maxFwdSpeed); // Plug angle into turning PID and get the resultant speed
                if(!finalFwdSpeed) finalFwdSpeed = 2;
                if(!fwd) finalFwdSpeed = -finalFwdSpeed;
                VRUN(finalFwdSpeed, finalFwdSpeed);

                task::sleep(5);
            }
            VRUN(0, 0);
            fwdControl->reset();
        }

        
    };
};