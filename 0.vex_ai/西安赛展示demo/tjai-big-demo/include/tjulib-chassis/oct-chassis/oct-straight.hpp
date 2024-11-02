#pragma once    
#include "tjulib-chassis/oct-chassis/oct-base.hpp"


extern double zero_drift_error;

namespace tjulib
{
    using namespace vex;

    class Oct_StraChassis : virtual public Oct_BaseChassis {
    protected:
        pidControl* fwdControl = NULL;      // 直线移动pid控制�?   
        pidControl* turnControl = NULL;     // �?向pid控制�?
    private:
        const double PI = 3.14159265358979323846;

    public:
        Oct_StraChassis(std::vector<std::vector<vex::motor*>*>& _chassisMotors, pidControl* _motorpidControl, Position* _position, const T _r_motor, pidControl* _fwdpid, pidControl* _turnpid) :
            Oct_BaseChassis(_chassisMotors, _motorpidControl, _position, _r_motor),
            fwdControl(_fwdpid), turnControl(_turnpid) {}

        /* ============== 打点控制直线平移, 方向0~360°, deg ===============*/
        void simpleMove(T speed, T angle, T sec, T gaptime = 10) {
            timer time;
            time.clear();
            T _speed = (speed / 100) * 850; // 850�?默�?�的电机最大转速inches/s
            while (1) {
                if (time.time() >= sec * 1000) {
                    break;
                }
                T v_X = _speed * sin((angle / 180) * PI);
                T v_Y = _speed * cos((angle / 180) * PI);
                // 计算每个�?子的速度
                T v_lf = v_Y + v_X;
                T v_lb = v_Y - v_X;
                T v_rf = -v_Y + v_X;
                T v_rb = -v_Y - v_X;
                VRUNStable(v_lf, v_lb, v_rf, v_rb);
                task::sleep(gaptime);
            }
        }


        /* ============== pid控制�?�? ===============*/
        void turnToAngle(double angle, T maxSpeed, double maxtime_ms, int fwd = 1, int back = 0) {
            timer mytime;
            mytime.clear();
            double totaltime = 0;
            T finalTurnSpeed = 20;

            double targetDeg = Math::getWrap360(angle); // Obtain the closest angle to the target position
            // 调转车头方向
            if(back){
                targetDeg += 180;
            }
            targetDeg = Math::getWrap360(targetDeg);

            double currentAngle = Math::getWrap360(imu.rotation());

            double prev_speed = finalTurnSpeed;

            int init = 0;

            T error = optimalTurnAngle(targetDeg, currentAngle);

            turnControl->resetpid();

            while (!turnControl->overflag() || (fabs(error) >= 2)) // If within acceptable distance, PID output is zero.
            {

                if (totaltime = mytime.time(msec) >= maxtime_ms) {
                    break;
                }
                if (std::fabs(error) < turnControl->params->errorThreshold && finalTurnSpeed <= turnControl->params->minSpeed) {
                    turnControl->cnt++;
                }

                // 大小角调�?
                currentAngle = imu.angle() - zero_drift_error;
                // 计算error
                if (fwd)
                    error = optimalTurnAngle(targetDeg, currentAngle);
                else
                    error = targetDeg - currentAngle;

                finalTurnSpeed = turnControl->pidCalcu(error, maxSpeed); // Plug angle into turning PID and get the resultant speed

                if (finalTurnSpeed * prev_speed < 0 && init > 0) {
                    maxSpeed *= 0.3;
                }
                init = 1;

                prev_speed = finalTurnSpeed;

                VRUN(finalTurnSpeed, finalTurnSpeed, finalTurnSpeed, finalTurnSpeed);
                //printf("error:%lf, finalTurnSpeed:%lf\n",error, finalTurnSpeed);
                task::sleep(10);
            }

            turnControl->resetpid();

            VRUN(0, 0, 0, 0);
            setStop(vex::brakeType::brake);
        }

        /* ============== pid控制�?向目标点 ===============*/
        void turnToTarget(Point target, T maxSpeed, double maxtime_ms, int fwd = 1, int back = 0){
            T deg = 90 - getDegree(target);
            if (deg < 0)
                deg += 360;
            turnToAngle(deg, maxSpeed, maxtime_ms, fwd, back);
        }

        

        /* ============== pid控制平移向目标点, 不能控制终态�?�度 ===============*/
        void moveToTarget(Point target, T maxSpeed = 100, T maxtime_ms = 5000, T gaptime = 10, int fwd = 1) {
            timer mytime;
            mytime.clear();

            T finalSpeed = 20;

            T current_distance = GetDistance(target);   // 距�?�目标点的距�?
            T current_localAngle = getLocalDegree(target); // �?向差�?(deg)
            fwdControl->resetpid();
            while (!fwdControl->overflag()) {
                current_distance = GetDistance(target);
                current_localAngle = getLocalDegree(target);
                
                //printf("targetDistant: %lf  current_localAngle: %lf this_angle: %lf\n  ", current_distance, current_localAngle, position->globalPoint.angle);
                //printf("position->globalPoint.x: %lf, position->globalPoint.y : %lf\n",position->globalPoint.x ,position->globalPoint.y );
                if (current_distance <= fwdControl->params->errorThreshold) {
                    fwdControl->cnt++;
                }
                if (mytime.time(msec) >= maxtime_ms) {
                    break;
                }
                finalSpeed = fwdControl->pidCalcu(current_distance, maxSpeed);

                if (!fwd) finalSpeed = -finalSpeed;

                T fwdSpeed_y = finalSpeed * cos(current_localAngle / 180 * PI);
                T fwdSpeed_x = finalSpeed * sin(current_localAngle / 180 * PI);

               // printf("fwdSpeed_x: %lf  fwdSpeed_y: %lf", fwdSpeed_x, fwdSpeed_y);
                // 计算每个�?子的速度
                T fwdSpeed_lf = fwdSpeed_y + fwdSpeed_x;
                T fwdSpeed_lb = fwdSpeed_y - fwdSpeed_x;
                T fwdSpeed_rf = -fwdSpeed_y + fwdSpeed_x;
                T fwdSpeed_rb = -fwdSpeed_y - fwdSpeed_x;

                VRUN(fwdSpeed_lf, fwdSpeed_lb, fwdSpeed_rf, fwdSpeed_rb);
                task::sleep(gaptime);
            }
            VRUN(0,0,0,0);
            fwdControl->resetpid();
        }

        /* ============== pid控制直线平移, 方向0~360°, deg ===============*/
        void moveInches(T inches, T fwdAngle, T maxSpeed, T maxtime_ms = 5000, T gaptime = 10, int fwd = 1) {

            timer mytime;
            mytime.clear();
            T finalFwdSpeed = 20;
            T targetDistant = inches;
            

            //�?标位�?计算
            T target_y = position->globalPoint.y + targetDistant * cos((fwdAngle / 180) * PI);
            T target_x = position->globalPoint.x + targetDistant * sin((fwdAngle / 180) * PI);

            moveToTarget({target_x, target_y, position->globalPoint.angle}, maxSpeed, maxtime_ms, gaptime, fwd);

            
        }


        /* ============== pid控制一边直线�?�走一边转�?, 能控制终态�?�度 target.angle : deg ===============*/
        void RotMoveToTarget(Point target, T maxSpeed = 100, T maxtime_ms = 5000, T gaptime = 10, int fwd = 1) {
            timer mytime;
            mytime.clear();
            T Speed = 20;

            T current_distance = GetDistance(target);   // 距�?�目标点的距�?
            T current_localAngle = getLocalDegree(target); // �?向差�?(deg)

            T initial_error_angle = optimalTurnAngle(target.angle, position->globalPoint.angle / PI * 180);  
            T current_error_angle = initial_error_angle;
            // 重置pid控制�?(事实上这里并没有用到turnControl->params的Kp,Ki,Kp参数，只�?利用了turnControl�?的�??�?允值及震荡cnt)
            fwdControl->resetpid();
            turnControl->resetpid();

            while(!fwdControl->overflag() || !turnControl->overflag()){ // 必须�?向大�?

                // 每�?�循�?都需要更新一下距离目标点的距离以及航向差�?
                current_distance = GetDistance(target);
                current_localAngle = getLocalDegree(target);    // deg
                current_error_angle = optimalTurnAngle(target.angle, position->globalPoint.angle / PI * 180);

                // 终�?�条件判�?
               
                if (fabs(current_distance) <= fwdControl->params->errorThreshold) {
                    
                    fwdControl->cnt++;
                }
                if(fabs(current_error_angle) <= turnControl->params->errorThreshold){
                    turnControl->cnt++;
                }
                if (mytime.time(msec) >= maxtime_ms) {
                    break;
                }
                
                /*=============================================== �?角底盘四�?控制 ==============================================
                
                        左前�? v_lf =  ( sqrt(2)/2 )*v*( 1/(1-m) )*( sin(θ)+cos(θ) )
                        右后�? v_rb =  ( sqrt(2)/2 )*v*( m/(1-m) )*( sin(θ)+cos(θ) )
                        右前�? v_rf =  ( sqrt(2)/4 )*v*(sin(θ)-cos(θ)) + 0.5*( γ-( sqrt(2)/2 )*( (1+m)/(1-m) )*( sin(θ)+cos(θ) )
                        左后�? v_lb = -( sqrt(2)/4 )*v*(sin(θ)-cos(θ)) + 0.5*( γ-( sqrt(2)/2 )*( (1+m)/(1-m) )*( sin(θ)+cos(θ) )
                
                        控制思想 : 
                        该问题需要控制两�?�?由度，即距�?�目标点的距离d和在机器人自己相对坐标系下与�?标点的y轴�?�方向夹角�?
                        这里构建平动、转动约束，平动直接用v分解列方程（x轴方向一�?，y轴方向一�?），�?动是等于γ*v一�?，也就是利用γ控制�?动转速，
                        这里规定旋转权γ为(-1, 1)的参数，使用p控制器思想计算
                        但是现在�?有三�?约束方程，因此�?�为添加一�?约束v_lf = m*v_rb，这样可以带着m求解最终结�?
                        不妨�?m=-1, 这样�?以解决保持电压输出的稳定，同时式子可以退化到一�?很简单的形式，等价于右前和左后轮添加了一�?旋转�?正项
                
                =============================================================================================================*/

                // 根据距�?�pid计算平移速度
                Speed = fwdControl->pidCalcu(current_distance, maxSpeed);
                T fwdSpeed_y = Speed * cos(current_localAngle / 180 * PI);
                T fwdSpeed_x = Speed * sin(current_localAngle / 180 * PI);
                // 计算旋转�?
                T gamma = current_error_angle / fabs(initial_error_angle);  // 旋转权�?�算需要保留�?�号
                // gamma最小限�?
                if(fabs(gamma) < 0.2){
                    gamma = 0.2 * current_error_angle / fabs(current_error_angle);
                }
                // 当Initial角绝对值太小的时候会出现过震荡，显然�?不合理的
                if(fabs(initial_error_angle) < 10){
                    gamma = ( current_error_angle / fabs(current_error_angle) )* 0.1;
                }
                // 达到范围阈值了就暂停移动调整，�?以依靠最后到达位�?的pid调整
                if(fabs(current_error_angle) <= turnControl->params->errorThreshold){
                    gamma = 0;
                }
               // printf("gamma : %lf \n", gamma);
               // printf("current_error_angle : %lf initial_error_angle : %lf\n", current_error_angle, initial_error_angle);
               // printf("rotation : %lf \n", 2 * gamma * fabs(fwdSpeed_y + fwdSpeed_x ));
                //printf("targetDistant: %lf, current_localAngle : %lf\n", current_distance, current_localAngle);
                //printf("fwdControl->cnt: %d turnControl->cnt: %d\n", fwdControl->cnt, turnControl->cnt);
                //printf("rotation : %lf\n", 0.5 * Speed * gamma * ( sin(current_localAngle / 180 * PI) + cos(current_localAngle / 180 * PI) ));
                
                // 计算四轮速度
                T v_lf =  fwdSpeed_y + fwdSpeed_x ;
                T v_rb =  -fwdSpeed_y - fwdSpeed_x;
                T v_rf =  -fwdSpeed_y + fwdSpeed_x +  2 * gamma * fabs(fwdSpeed_y + fwdSpeed_x );
                T v_lb = fwdSpeed_y - fwdSpeed_x +  2 * gamma * fabs(fwdSpeed_y + fwdSpeed_x );
                //printf("fwdControl->cnt: %d turnControl->cnt: %d\n", fwdControl->cnt, turnControl->cnt);
                
                // 考虑如果到达位置但是没有达到�?向�?�求，按照转向去处理
                if(fwdControl->overflag() && !turnControl->overflag()){
                    T finalTurnSpeed = turnControl->pidCalcu(current_error_angle, maxSpeed);
                    v_lf = finalTurnSpeed, v_rb = finalTurnSpeed, v_rf = finalTurnSpeed, v_lb = finalTurnSpeed;
                   // printf("23232323232\n");
                }
                
                // 输出四轮控制电压
                VRUN(v_lf, v_lb, v_rf, v_rb);
                //printf("v_lf : %lf v_rb : %lf v_rf : %lf v_lb : %lf \n", v_lf, v_rb, v_rf, v_lb);
                task::sleep(gaptime);

            }
        }
        void SetXMove(T x, T maxSpeed, double maxtime_ms = 5000, int fwd = 1){
            timer mytime;
            mytime.clear();
            T finalFwdSpeed = 20;
            T target = x;
            T startError = target - GPS_.xPosition(inches);
            fwdControl->resetpid();
           
            while (!fwdControl->overflag()) // If within acceptable distance, PID output is zero.
            {
                if(fabs(startError)<= 2 && finalFwdSpeed <= 15){
                    fwdControl->cnt++;
                }

                // printf("cnt %lf \n", fwdControl->cnt);
                // printf("speed %lf \n", finalFwdSpeed);
                startError = (target - GPS_.xPosition(inches)) * 0.15; // Obtain the closest angle to the target position

                if(mytime.time(msec)>=maxtime_ms){
                    break;
                }
                finalFwdSpeed = fwdControl->pidCalcu(startError * 0.15, maxSpeed); // Plug angle into turning PID and get the resultant speed 
                
                if(!finalFwdSpeed) finalFwdSpeed = 2;
                if(!fwd) finalFwdSpeed = -finalFwdSpeed;
                VRUN(finalFwdSpeed, finalFwdSpeed, -finalFwdSpeed, -finalFwdSpeed);

                task::sleep(30);
            }
            VRUN(0, 0, 0, 0);
            fwdControl->resetpid();
        }
        // 基于距�?�传感器读数的pid
        void DistanceSensorMove(T mms, T maxSpeed, double maxtime_ms = 5000, int fwd = 1){
            timer mytime;
            mytime.clear();
            T finalFwdSpeed = 20;
            T targetDistant = mms;
            T startError = DistanceSensor.objectDistance(mm);
            fwdControl->resetpid();
           
            while (!fwdControl->overflag()) // If within acceptable distance, PID output is zero.
            {
                if(targetDistant<=0.3 && finalFwdSpeed <= 15){
                    fwdControl->cnt++;
                }
                  printf("error: %lf distance: %lf finalFwdSpeed:%lf\n", targetDistant, DistanceSensor.objectDistance(mm), finalFwdSpeed);
                // printf("cnt %lf \n", fwdControl->cnt);
                // printf("speed %lf \n", finalFwdSpeed);
                targetDistant = -(mms - fabs(DistanceSensor.objectDistance(mm)) ) * 0.1; // Obtain the closest angle to the target position
                
                if(mytime.time(msec)>=maxtime_ms){
                    break;
                }
                finalFwdSpeed = fwdControl->pidCalcu(targetDistant, maxSpeed); // Plug angle into turning PID and get the resultant speed 
                
                if(!finalFwdSpeed) finalFwdSpeed = 2;
                if(!fwd) finalFwdSpeed = -finalFwdSpeed;
                VRUN(finalFwdSpeed, finalFwdSpeed, -finalFwdSpeed, -finalFwdSpeed);

                task::sleep(10);
            }
            VRUN(0, 0, 0, 0);
            fwdControl->resetpid();
        }

        // Vision�?向移动�?�齐(在�?�固定桩的时候使�?)
        void VisionSensorMove(T x, T maxSpeed, double maxtime_ms = 5000, int fwd = 1){

             timer mytime;
            mytime.clear();
            T finalFwdSpeed = 20;
            T target = x;
            int error = target - Vision_front.largestObject.centerX;
            fwdControl->resetpid();
           
            while (!fwdControl->overflag()) // If within acceptable distance, PID output is zero.
            {
                if(error<=0.5 && finalFwdSpeed <= 15){
                    fwdControl->cnt++;
                }

                // printf("cnt %lf \n", fwdControl->cnt);
                // printf("speed %lf \n", finalFwdSpeed);
                
                if(mytime.time(msec)>=maxtime_ms){
                    break;
                }
                int see_flag = 0;
                Vision_front.takeSnapshot(Stake_Red);
                if(Vision_front.largestObject.exists){
                    error = target - Vision_front.largestObject.centerX;
                    error *= 0.1;
                    finalFwdSpeed = fwdControl->pidCalcu(error, maxSpeed); // Plug angle into turning PID and get the resultant speed 
                    see_flag = 1;
                }
                Vision_front.takeSnapshot(Stake_Blue);
                if(Vision_front.largestObject.exists){
                    error = target - Vision_front.largestObject.centerX;
                    error *= 0.2;
                    finalFwdSpeed = fwdControl->pidCalcu(error, maxSpeed); // Plug angle into turning PID and get the resultant speed 
                    see_flag = 1;
                }
                Vision_front.takeSnapshot(Stake_Yellow);
                if(Vision_front.largestObject.exists){
                    error = target - Vision_front.largestObject.centerX;
                    error *= 0.2;
                    finalFwdSpeed = fwdControl->pidCalcu(error, maxSpeed); // Plug angle into turning PID and get the resultant speed
                    see_flag = 1;
                }
                if(!see_flag){
                    break;
                }
                if(!finalFwdSpeed) finalFwdSpeed = 2;
                if(!fwd) finalFwdSpeed = -finalFwdSpeed;
                VRUN(finalFwdSpeed, -finalFwdSpeed, finalFwdSpeed, -finalFwdSpeed);

                task::sleep(30);
            }
            VRUN(0, 0, 0, 0);
            fwdControl->resetpid();


        }

       

    };
};