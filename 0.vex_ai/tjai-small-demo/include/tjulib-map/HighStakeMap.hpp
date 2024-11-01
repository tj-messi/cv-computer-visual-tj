#pragma once
#include "vex.h"
#include "tjulib-position/PositionStrategy.hpp"

typedef double T;

namespace tjulib
{
    // 障碍物的类型
    struct Obstacle {
        T x;
        T y;
        T radius;
        int type = 0; // 0代表默认（不关心类型）， 1代表杆子， 2代表桩, 3代表固定桩 （这里根据需要可以随便设置意义）
        // 默认构造函数
        Obstacle() : x(0), y(0), radius(0), type(0) {}
        Obstacle(T x, T y, T radius, int type) : x(x), y(y), radius(radius), type(type) {}
    };

    struct Ring{
        T x;
        T y;
        int color;   //  0为红, 蓝为1
        // 默认构造函数
        Ring() : x(0), y(0), color(0) {}
    };
    
    // 环类型
    class HighStakeMap
    {
        

    protected:
        Position *position = NULL;
        const T PI = 3.14159265358979323846;

    public:
        // 障碍物构成的列表
        std::vector<Obstacle> obstacleList;
        std::vector<Obstacle> stakesList;
        std::vector<Obstacle> ringsList;
        std::vector<Obstacle> fixedStakes;     // 固定桩
        // 加分区
        std::vector<Obstacle> bonusAreas;
    public:
        // 构造函数
        HighStakeMap(Position* _position) :  position(_position) {
            // 加分区坐标
            bonusAreas = {{59, 59, 5, 3}, {-59, 59, 5, 3}, {59, -59, 5, 3}, {-59, -59, 5, 3}};
            // 固定桩坐标
            fixedStakes = {{60, 0, 12, 3}, {-60, 0, 12, 3}, {0, -60, 12, 3}, {0, 60, 12, 3}};
            // 构造时初始化地图杆子障碍物
            obstacleList = {{-24, 0, 3, 1}, {24, 0, 3, 1}, {0, 24, 3, 1}, {0, -24, 3, 1}};
            // 初始化环和桩的先验坐标
        }

    };
}