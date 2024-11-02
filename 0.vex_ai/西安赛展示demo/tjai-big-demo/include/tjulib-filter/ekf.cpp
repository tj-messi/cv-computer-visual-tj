#include "ekf.hpp"
#include "robot-config.h"
//sys函数
Eigen::Vector3d model(const Eigen::Vector3d xt_1, const Eigen::Vector3d u){
    Eigen::Vector3d xtt_1;
    double global_theta = xt_1(2) + (u(2) / 2) - 3.1415926 * encoder_rotate_degree / 180; // 这里是编码器45°装置的，如果90°装配则需要变化
    // x(k) = x(k-1) + ROT(I+OFF)*u(k-1)
    xtt_1(0) = xt_1(0) + cos(global_theta) * u(0) + sin(global_theta) * u(1) + (hOffset * sin(global_theta) + vOffset * cos(global_theta)) * u(2);
    xtt_1(1) = xt_1(1) - sin(global_theta) * u(0) + cos(global_theta) * u(1) + (hOffset * cos(global_theta) - vOffset * sin(global_theta)) * u(2);
    xtt_1(2) = xt_1(2) + u(2);
    return xtt_1;
}

Eigen::Matrix3d F_jac(const Eigen::Vector3d& xtt_1, const Eigen::Vector3d& u) {
    Eigen::MatrixXd result(3, 3);
    result << 1, 0, 0,
              0, 1, 0,
              0, 0, 1;
    return result;
}

Eigen::Vector3d h(const Eigen::Vector3d& x){
    Eigen::Vector3d z= H_jac(x)* x;
    return z;
}

Eigen::Matrix3d H_jac(const Eigen::Vector3d& xtt_1) {
    Eigen::Matrix3d H;
    H << 1, 0, 0,
         0, 1, 0,
         0, 0, 1;
    return H;
}
