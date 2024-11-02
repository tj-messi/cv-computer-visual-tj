#ifndef EKF_HPP
#define EKF_HPP
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#undef __ARM_NEON__
#undef __ARM_NEON

#include "eigen/Eigen/Dense"
// 初始化各个矩阵
Eigen::Vector3d model(const Eigen::Vector3d xt_1, const Eigen::Vector3d u);
Eigen::Vector3d h(const Eigen::Vector3d& x);
Eigen::Matrix3d F_jac(const Eigen::Vector3d& xtt_1, const Eigen::Vector3d& u);
Eigen::Matrix3d H_jac(const Eigen::Vector3d& xtt_1);

class ExtendedKalmanFilter {
private:
    std::function<Eigen::Vector3d(const Eigen::Vector3d&, const Eigen::Vector3d&)> f_;
    std::function<Eigen::Vector3d(const Eigen::Vector3d&)> h_;
    std::function<Eigen::Matrix3d(const Eigen::Vector3d&, const Eigen::Vector3d&)> F_jac_;
    std::function<Eigen::Matrix3d(const Eigen::Vector3d&)> H_jac_;
    Eigen::Matrix3d Q_, R_;
    Eigen::Vector3d x_;
    Eigen::MatrixXd P_;

public:
    ExtendedKalmanFilter(
        std::function<Eigen::Vector3d(const Eigen::Vector3d&, const Eigen::Vector3d&)> f=model,
        std::function<Eigen::Vector3d(const Eigen::Vector3d&)> h1=h,
        std::function<Eigen::Matrix3d(const Eigen::Vector3d&, const Eigen::Vector3d&)> F_jac1=F_jac,
        std::function<Eigen::Matrix3d(const Eigen::Vector3d&)> H_jac1 = H_jac
        
    ) : f_(f), h_(h1), F_jac_(F_jac1), H_jac_(H_jac1), P_(Eigen::MatrixXd::Constant(3,3,0)) {
        Q_ << 1,0,0,
            0,1,0,
            0,0,0.15;
        R_ << 0.7,0,0,
            0,0.7,0,
            0,0,0.15;
        x_ << 0, 0, 0;
    }

    ExtendedKalmanFilter(
        const Eigen::Matrix3d& Q,
        const Eigen::Matrix3d& R,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f=model,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h1=h,
        std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> F_jac1=F_jac,
        std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H_jac1 = H_jac
        
    ) : Q_(Q),R_(R),f_(f), h_(h1), F_jac_(F_jac1), H_jac_(H_jac1), P_(Eigen::MatrixXd::Constant(3,3,0)) {
        
    }

    void setgains(
        const Eigen::Matrix3d& Q,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& x0
    ){
        Q_=(Q), R_=(R), x_=(x0);
    }

    void setx0(const double& x, const double& y){
        x_(0) = x;
        x_(1) = y;
    }

    void set_QR(const double Q_config, const double R_config){

        const double Q_config_ = Q_config;
        const double R_config_ = R_config;
        Q_ << Q_config_, 0, 0,
            0, Q_config_, 0,
            0, 0, 0.15;
        R_ << R_config_, 0, 0,
            0, R_config_, 0,
            0, 0, 0.15;
    }

    void predict(const Eigen::Vector3d& xt_1, const Eigen::Vector3d& u) {
        x_ = f_(xt_1, u);           // 计算估测方程，根据(k - 1)次的观测状态以及输入量计算第k次的的估测量
        Eigen::MatrixXd F = F_jac_(x_,u);   
        P_ = F * P_ * F.transpose() + Q_;   // 计算误差的协方差矩阵
    }

    void update(const Eigen::Vector3d& z) {
        Eigen::MatrixXd H = H_jac_(x_);        // H矩阵
        Eigen::VectorXd y = z - h_(x_);   // 观测结果带来的信息增益
        // 计算第k次的卡尔曼增益K(k)
        Eigen::MatrixXd S = H * P_ * H.transpose() + R_;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
        // 数据融合
        x_ = x_ + K * y;
        // 更新误差的协方差矩阵
        P_ = (Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K * H) * P_;
    }

    const Eigen::Vector3d getState() {
        return this->x_;
    }
};

#endif
 