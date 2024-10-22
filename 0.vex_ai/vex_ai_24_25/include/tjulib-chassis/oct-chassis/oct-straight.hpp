#pragma once    
#include "tjulib-chassis/oct-chassis/oct-base.hpp"


extern double zero_drift_error;

namespace tjulib
{
    using namespace vex;

    class Oct_StraChassis : virtual public Oct_BaseChassis {
    protected:
        pidControl* fwdControl = NULL;      // 直线移动pid控制器   
        pidControl* turnControl = NULL;     // 转向pid控制器
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
            T _speed = (speed / 100) * 850; // 850是默认的电机最大转速inches/s
            while (1) {
                if (time.time() >= sec * 1000) {
                    break;
                }
                T v_X = _speed * sin((angle / 180) * PI);
                T v_Y = _speed * cos((angle / 180) * PI);
                // 计算每个轮子的速度
                T v_lf = v_Y + v_X;
                T v_lb = v_Y - v_X;
                T v_rf = -v_Y + v_X;
                T v_rb = -v_Y - v_X;
                VRUNStable(v_lf, v_lb, v_rf, v_rb);
                task::sleep(gaptime);
            }
        }


        /* ============== pid控制转向 ===============*/
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

                // 大小角调整
                currentAngle = imu.angle() - zero_drift_error;
                // 计算error
                if (fwd)
                    error = optimalTurnAngle(targetDeg, currentAngle);
                else
                    error = targetDeg - currentAngle;

                finalTurnSpeed = turnControl->pidCalcu(error, maxSpeed); // Plug angle into turning PID and get the resultant speed

                if (finalTurnSpeed * prev_speed < 0 && init > 0) {
                    maxSpeed *= 0.2;
                }
                init = 1;

                prev_speed = finalTurnSpeed;

                VRUN(finalTurnSpeed, finalTurnSpeed, finalTurnSpeed, finalTurnSpeed);

                task::sleep(5);
            }

            turnControl->resetpid();

            VRUN(0, 0, 0, 0);
            setStop(vex::brakeType::brake);
        }

        /* ============== pid控制转向目标点 ===============*/
        void turnToTarget(Point target, T maxSpeed, double maxtime_ms, int fwd = 1, int back = 0){
            T deg = 90 - getDegree(target);
            if (deg < 0)
                deg += 360;
            turnToAngle(deg, maxSpeed, maxtime_ms, fwd, back);
        }

        

        /* ============== pid控制平移向目标点, 不能控制终态角度 ===============*/
        void moveToTarget(Point target, T maxSpeed = 100, T maxtime_ms = 5000, T gaptime = 10, int fwd = 1) {
            timer mytime;
            mytime.clear();

            T finalSpeed = 20;

            T current_distance = GetDistance(target);   // 距离目标点的距离
            T current_localAngle = getLocalDegree(target); // 航向差角(deg)
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
                // 计算每个轮子的速度
                T fwdSpeed_lf = fwdSpeed_y + fwdSpeed_x;
                T fwdSpeed_lb = fwdSpeed_y - fwdSpeed_x;
                T fwdSpeed_rf = -fwdSpeed_y + fwdSpeed_x;
                T fwdSpeed_rb = -fwdSpeed_y - fwdSpeed_x;

                VRUN(fwdSpeed_lf, fwdSpeed_lb, fwdSpeed_rf, fwdSpeed_rb);
                task::sleep(gaptime);
            }
            fwdControl->resetpid();
        }

        /* ============== pid控制直线平移, 方向0~360°, deg ===============*/
        void moveInches(T inches, T fwdAngle, T maxSpeed, T maxtime_ms = 5000, T gaptime = 10, int fwd = 1) {

            timer mytime;
            mytime.clear();
            T finalFwdSpeed = 20;
            T targetDistant = inches;
            

            //目标位置计算
            T target_y = position->globalPoint.y + targetDistant * cos((fwdAngle / 180) * PI);
            T target_x = position->globalPoint.x + targetDistant * sin((fwdAngle / 180) * PI);

            moveToTarget({target_x, target_y, position->globalPoint.angle}, maxSpeed, maxtime_ms, gaptime, fwd);

            
        }


        /* ============== pid控制一边直线行走一边转向, 能控制终态角度 target.angle : deg ===============*/
        void RotMoveToTarget(Point target, T maxSpeed = 100, T maxtime_ms = 5000, T gaptime = 10, int fwd = 1) {
            timer mytime;
            mytime.clear();
            T Speed = 20;

            T current_distance = GetDistance(target);   // 距离目标点的距离
            T current_localAngle = getLocalDegree(target); // 航向差角(deg)

            T initial_error_angle = optimalTurnAngle(target.angle, position->globalPoint.angle / PI * 180);  
            T current_error_angle = initial_error_angle;
            // 重置pid控制器(事实上这里并没有用到turnControl->params的Kp,Ki,Kp参数，只是利用了turnControl中的误差允值及震荡cnt)
            fwdControl->resetpid();
            turnControl->resetpid();

            while(!fwdControl->overflag() || !turnControl->overflag()){ // 必须转向大表

                // 每次循环都需要更新一下距离目标点的距离以及航向差角
                current_distance = GetDistance(target);
                current_localAngle = getLocalDegree(target);    // deg
                current_error_angle = optimalTurnAngle(target.angle, position->globalPoint.angle / PI * 180);

                // 终止条件判定
               
                if (fabs(current_distance) <= fwdControl->params->errorThreshold) {
                    
                    fwdControl->cnt++;
                }
                if(fabs(current_error_angle) <= turnControl->params->errorThreshold){
                    turnControl->cnt++;
                }
                if (mytime.time(msec) >= maxtime_ms) {
                    break;
                }
                
                /*=============================================== 八角底盘四轮控制 ==============================================
                
                        左前轮 v_lf =  ( sqrt(2)/2 )*v*( 1/(1-m) )*( sin(θ)+cos(θ) )
                        右后轮 v_rb =  ( sqrt(2)/2 )*v*( m/(1-m) )*( sin(θ)+cos(θ) )
                        右前轮 v_rf =  ( sqrt(2)/4 )*v*(sin(θ)-cos(θ)) + 0.5*( γ-( sqrt(2)/2 )*( (1+m)/(1-m) )*( sin(θ)+cos(θ) )
                        左后轮 v_lb = -( sqrt(2)/4 )*v*(sin(θ)-cos(θ)) + 0.5*( γ-( sqrt(2)/2 )*( (1+m)/(1-m) )*( sin(θ)+cos(θ) )
                
                        控制思想 : 
                        该问题需要控制两个自由度，即距离目标点的距离d和在机器人自己相对坐标系下与目标点的y轴正方向夹角θ
                        这里构建平动、转动约束，平动直接用v分解列方程（x轴方向一个，y轴方向一个），转动是等于γ*v一个，也就是利用γ控制转动转速，
                        这里规定旋转权γ为(-1, 1)的参数，使用p控制器思想计算
                        但是现在只有三个约束方程，因此认为添加一个约束v_lf = m*v_rb，这样可以带着m求解最终结果
                        不妨令m=-1, 这样可以解决保持电压输出的稳定，同时式子可以退化到一个很简单的形式，等价于右前和左后轮添加了一个旋转修正项
                
                =============================================================================================================*/

                // 根据距离pid计算平移速度
                Speed = fwdControl->pidCalcu(current_distance, maxSpeed);
                T fwdSpeed_y = Speed * cos(current_localAngle / 180 * PI);
                T fwdSpeed_x = Speed * sin(current_localAngle / 180 * PI);
                // 计算旋转权
                T gamma = current_error_angle / fabs(initial_error_angle);  // 旋转权计算需要保留符号
                // gamma最小限制
                if(fabs(gamma) < 0.2){
                    gamma = 0.2 * current_error_angle / fabs(current_error_angle);
                }
                // 当Initial角绝对值太小的时候会出现过震荡，显然是不合理的
                if(fabs(initial_error_angle) < 10){
                    gamma = ( current_error_angle / fabs(current_error_angle) )* 0.1;
                }
                // 达到范围阈值了就暂停移动调整，可以依靠最后到达位置的pid调整
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
                
                // 考虑如果到达位置但是没有达到转向要求，按照转向去处理
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




    };
};