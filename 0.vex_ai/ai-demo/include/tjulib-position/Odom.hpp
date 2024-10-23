#pragma once

// 使用卡尔曼滤波减少误差则define，否则注释掉
#define KalmanFilter
#include "Math-Functions.h"
#include "vex.h"
#include "Position.hpp"

#ifdef KalmanFilter
#include"tjulib-filter/ekf.hpp"
#include"tjulib-filter/ekf.cpp"
#endif

namespace tjulib{
    
    using namespace vex;
    T prevtheta = 0;
    T theta1, deltaTheta=0.0;
    T V = 0.0; 

    #ifdef KalmanFilter
    //滤波器
    ExtendedKalmanFilter ekfilter;
    #endif

    struct encoderType{
        float vertical, horizonal;
    };

    class Odom : public Position{
    private:
        //Math
        Math OMath;
        // 轮子中心相对于旋转中心的偏移量(作垂线)
        const T horizonalOffset; //inches
        const T verticalOffset;
        const T rotate_degree;
        const T r_wheel_encoder;
        Point localDeltaPoint{0,0,0}; //change in x, change in y
        Point globalDeltaPoint{0,0,0}; //change in x, change in y

        encoder& encoderVertical;
        encoder& encoderHorizonal;
        inertial& imu;
        //SENSOR VALUES
        //encoder
        encoderType encoderVal; //verticalEnc, horizonalEnc, backEnc
        encoderType prevEncoderVal; //prev verticalEnc, horizonalEnc, backEnc
        encoderType deltaEncoderVal; //change in verticalEnc, horizonalEnc, backEnc
        //angle
        float currentAngle;
        float prevAngle;
        float deltaAngle; //rad
        const int PI = 3.1415926;

        

    public: 
        // kalmanfilter config
        Point KalmanFilterglobalPoint{0,0,0};  // kalman filter processed x, y
        int gps_drift_flag = 0;
        Odom(T hOffset, T vOffset, T r_wheel_encoder, T rotate_degree, encoder& encoderVertical, encoder& encoderHorizonal, inertial& imu):
        horizonalOffset(hOffset), verticalOffset(vOffset), OMath(r_wheel_encoder), rotate_degree(rotate_degree),
        encoderVertical(encoderVertical), encoderHorizonal(encoderHorizonal), imu(imu), r_wheel_encoder(r_wheel_encoder){
           // encoderVertical.resetRotation(); 
           // encoderHorizonal.resetRotation(); 
            globalPoint = {0, 0, 0};
            prevGlobalPoint = {0, 0, 0};
            globalDeltaPoint = {0, 0, 0};

            //LOCAL COORDINATES
            localDeltaPoint = {0, 0};

            //SENSOR VALUES
            //encoder
            encoderVal = {0, 0}; //verticalEnc, horizonalEnc, backEnc
            prevEncoderVal = {0, 0};
            deltaEncoderVal = {0, 0};
            //angle
            currentAngle = 0.0;
            prevAngle = 0.0;
            deltaAngle = 0.0;
        }
#ifdef KalmanFilter
        // ekf
        void filteringData(){
        T Q_config = 0.0;                      // Q_Matrix in configuration x, y
        T R_config = 0.0;                      // R_Matrix configuration in x, y
            if(gps_drift_flag){     // 当gps漂移的时候优先相信编码轮
                    Q_config = 0.1;
                    R_config = 6;
            }else{
                    Q_config = 1.5;
                    R_config = 1;
                    
            }
            // 设置Q、R噪声的方差矩阵参数
            ekfilter.set_QR(Q_config, R_config);
            Eigen::Vector3d u;      // 输入量
            Eigen::Vector3d xt_1;   // (k - 1)时刻状态量
            Eigen::Vector3d z;      // 观测量
            u << deltaEncoderVal.vertical, deltaEncoderVal.horizonal, globalDeltaPoint.angle;
            xt_1 << prevGlobalPoint.x, prevGlobalPoint.y, prevGlobalPoint.angle;
            // 预测
            ekfilter.predict(xt_1, u);
            z<<gps_x, gps_y, gps_heading;
            // 更新
            ekfilter.update(z);
            // 第(k-1) -> 第(k)次的计算完成
            Eigen::Vector3d xt = ekfilter.getState();
            globalPoint.x = xt(0);
            globalPoint.y = xt(1);
            //globalPoint.angle = xt(2);
           // KalmanFilterglobalPoint.x = xt(0);
           // KalmanFilterglobalPoint.y = xt(1);
           // KalmanFilterglobalPoint.angle = xt(2);
        }
#endif
        
        //ODOMETRY FUNCTIONS
        void updateSensors(){

            // 读取当前horizonal方向里程计的总转周（inches）
            encoderVal.horizonal = -OMath.degToInch(encoderHorizonal.rotation(deg)); //horizonalE
            encoderVal.vertical = OMath.degToInch(encoderVertical.rotation(deg)); //horizonalE
            
            // 计算编码器变化的delta
            deltaEncoderVal.vertical = (encoderVal.vertical - prevEncoderVal.vertical) ; //verticalE
            deltaEncoderVal.horizonal = (encoderVal.horizonal - prevEncoderVal.horizonal); //horizonalE
            //printf("encoderVal.horizonal : %lf \n", deltaEncoderVal.horizonal);
            //printf("encoderVal.vertical : %lf \n", deltaEncoderVal.vertical);
            // 更新
            prevEncoderVal.vertical = encoderVal.vertical; //verticalE
            prevEncoderVal.horizonal = encoderVal.horizonal; //horizonalE

            // 获取当前朝向
            currentAngle = OMath.getRadians(imu.rotation());
            currentAngle = Math::getWrap2pi(currentAngle);

            deltaAngle = currentAngle - prevAngle;
            prevAngle = currentAngle;

        }

        void updatePosition(){
            //Polar coordinates
            T deltaX = deltaEncoderVal.horizonal;
            T deltaY = deltaEncoderVal.vertical;
            T localX = 0;
            T localY = 0;

            if (deltaAngle == 0){ // prevent divide by 0
                localX = deltaX;
                localY = deltaY;
            }
            else{    
                localX = 2 * sin(deltaAngle / 2) * (deltaX / deltaAngle + verticalOffset);
                localY = 2 * sin(deltaAngle / 2) * (deltaY / deltaAngle + horizonalOffset);
            }            
            T  global_angle = prevAngle + deltaAngle / 2 - PI * rotate_degree / 180;

            //Cartesian coordinates
            globalDeltaPoint.x = (localY * sin(global_angle)) + (localX * cos(global_angle)); 
            globalDeltaPoint.y = ((localY * cos(global_angle)) - (localX * sin(global_angle)));
            globalDeltaPoint.angle = deltaAngle;

            globalPoint.x = globalDeltaPoint.x + prevGlobalPoint.x;
            globalPoint.y = globalDeltaPoint.y + prevGlobalPoint.y;
            globalPoint.angle = currentAngle;

            prevGlobalPoint.x = globalPoint.x;
            prevGlobalPoint.y = globalPoint.y;
    
            #ifdef KalmanFilter
            filteringData();
            #endif



            return;
        }

        //将机器人的位置和传感器值重置为初始状态
        void reset(){
            encoderVertical.resetRotation(); 
            encoderHorizonal.resetRotation(); 
            prevEncoderVal.vertical = 0.0; prevEncoderVal.horizonal = 0.0; 
            prevAngle = 0.0;
            deltaAngle = 0.0;
            prevGlobalPoint = {0,0,0};
            globalDeltaPoint = {0,0,0};
        }

        void setPosition(float newX, float newY, float newAngle = -114514) override{
            reset();
            
            prevGlobalPoint.x = newX;
            prevGlobalPoint.y = newY;
            globalPoint.x = newX;
            globalPoint.y = newY;

            if(fabs(newAngle + 114514)>1){
                
                prevAngle = newAngle;
                currentAngle = newAngle;
                globalPoint.angle = newAngle;

                imu.setRotation(newAngle * 180 / 3.14159, deg);
                imu.setHeading(newAngle * 180 / 3.14159, deg);
                deltaAngle = 0;
                
            }
            #ifdef KalmanFilter
            ekfilter.setx0(newX, newY);
            #endif

        }

        //ODOMETRY THREAD
        void OdomRun(Odom& odom){
            T LeftSideMotorPosition=0, RightSideMotorPosition = 0;
            T LastLeftSideMotorPosition=0, LastRightSideMotorPosition = 0;
            timer time;
            time.clear();
            
            // gps度数记录
            Point pregpsPoint = {gps_x, gps_y, gps_heading};

            while(true) { 
                // 更新电机编码器
                for(motor* m : _leftMotors) LeftSideMotorPosition += OMath.getRadians(m->position(deg))/_leftMotors.size();
                for(motor* m : _rightMotors) RightSideMotorPosition += OMath.getRadians(m->position(deg))/_rightMotors.size();
                T LeftSideDistance=(LeftSideMotorPosition - LastLeftSideMotorPosition)*r_motor; 
                T RightSideDistance=(RightSideMotorPosition-LastRightSideMotorPosition)*r_motor;
                LeftBaseDistance = LeftSideMotorPosition*r_motor; RightBaseDistance = RightSideMotorPosition*r_motor;
               // printf("delta angle : %lf \n", deltaAngle);
                // 更新轮式里程计
                odom.updateSensors();
                // 更新位置坐标
                odom.updatePosition();

                // gps读数采样
                if(fabs((int)(time.time(msec)) % 100 )<20){
                    if(fabs(pregpsPoint.x - gps_x) >  0.5|| fabs(pregpsPoint.y - gps_y) > 0.5 ){
                        gps_drift_flag = 1;
                    }else{
                        gps_drift_flag = 0;
                    }
                    pregpsPoint.x = gps_x;
                    pregpsPoint.y = gps_y;
                }

                // 超偏强制更新
                if(fabs(gps_x - globalPoint.x) > 2.75 || fabs(gps_y - globalPoint.y) > 2.75){
                    globalPoint.x = gps_x;
                    globalPoint.y = gps_y;
                }

               //printf("X: %lf, Y: %lf, Heading : %lf \n",odom.globalPoint.x ,odom.globalPoint.y, odom.globalPoint.angle * 180 / 3.14159);
                
               // printf("GPS_X : %lf GPS_Y : %lf, GPS_Heading : %lf \n",gps_x, gps_y, gps_heading );
                //printf("X_Kalman: %lf, Y_Kalman: %lf \n",KalmanFilterglobalPoint.x ,KalmanFilterglobalPoint.y);
                this_thread::sleep_for(10); 
            }
        }

        void executePosition() override{
            OdomRun(*this);
        }

    };
};

