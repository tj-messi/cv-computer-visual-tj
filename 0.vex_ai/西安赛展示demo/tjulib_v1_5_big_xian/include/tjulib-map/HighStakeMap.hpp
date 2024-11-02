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
        // 由AI感知获取到的地图元素信息
        AI_RECORD *local_map = NULL;
        const T PI = 3.14159265358979323846;

    public:

        // 障碍物构成的列表
        std::vector<Obstacle> obstacleList;
        std::vector<Obstacle> stakesList;
        std::vector<Obstacle> ringsList;
        std::vector<Obstacle> fixedStakes;     // 固定桩
        // 加分区
        std::vector<Obstacle> bonusAreas;
        std::vector<Point> Rings;
        std::vector<Point> Stakes;
    public:
        // 构造函数
        HighStakeMap(Position* _position, AI_RECORD *_local_map) :  position(_position), local_map(_local_map) {
            // 加分区坐标
            bonusAreas = {{62, 62, 5, 3}, {-62, 62, 5, 3}, {62, -62, 5, 3}, {-62, -62, 5, 3}};
            // 固定桩坐标
            fixedStakes = {{60, 0, 12, 3}, {-60, 0, 12, 3}, {0, -60, 12, 3}, {0, 60, 12, 3}};
            // 构造时初始化地图杆子障碍物
            obstacleList = {{-24, 0, 4, 1}, {24, 0, 4, 1}, {0, 24, 4, 1}, {0, -24, 4, 1}};
            // 初始化环和桩的先验坐标
            Rings = {{-24, -24}, {-48, -48}, {-48, 0}, {0, -48}, {-24, 24}, 
                    {-48, 48}, {0, 48}, {48,  24}, {48, 48}, {24, 48}, {48, -24}, 
                    {-24, -48}, {-48, -48}, {60, 0}, {0, 60}, {0, -60}};

            Stakes = {{-24, 48}, {24, 24}, {48, 0}, {24, -24}, {-24, -48}};
        }

        // 函数的更新坐标
        void UpdatePos(){
            // 检查桩的坐标更新
            for(auto &stake_post : local_map->detections){
                for(auto &stake_prior : Stakes){
                    // 更新桩的
                    if(sqrt((stake_post.mapLocation.x-stake_prior.x) * (stake_post.mapLocation.x-stake_prior.x) + (stake_post.mapLocation.x-stake_prior.y) * (stake_post.mapLocation.x-stake_prior.y))<2){
                        stake_prior.x = (stake_post.mapLocation.x + stake_prior.x) / 2;
                        stake_prior.y = (stake_post.mapLocation.y + stake_prior.y) / 2;
                    }
                }
                
            }
            // 检查环的坐标更新
            for(auto &ring_post : local_map->detections){
                for(auto &ring_prior : Rings){
                    // 更新桩的
                    if(sqrt((ring_post.mapLocation.x-ring_prior.x) * (ring_post.mapLocation.x-ring_prior.x) + (ring_post.mapLocation.x-ring_prior.y) * (ring_post.mapLocation.x-ring_prior.y))<2){
                        ring_prior.x = (ring_post.mapLocation.x + ring_prior.x) / 2;
                        ring_prior.y = (ring_post.mapLocation.y + ring_prior.y) / 2;
                    }
                }
                
            }
            task::sleep(200);
        }
    };
}