/* 
        该PurePursuit算法参考PURDUE SIGBOTS wiki开源库,
        交点计算公式推导与追踪点选择算法详见：
        https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit

*/
#pragma once
#include "vex.h"
#include <cmath>
#include "tjulib-position/PositionStrategy.hpp"

namespace tjulib
{
    using namespace vex;
    
class PurePursuit{
    protected:
        Position *position = NULL;
        const T PI = 3.14159265358979323846;

    public:
        PurePursuit(Position* _position) :  position(_position) {}

        T pt_to_pt_distance(Point pt1, Point pt2) {
            T distance = sqrt((pt2.x - pt1.x)*(pt2.x - pt1.x) + (pt2.y - pt1.y)*(pt2.y - pt1.y));
            return distance;
        }

        int sgn(double x) {
            return (x > 0) ? 1 : (x < 0) ? -1 : 0;
        }

        /*=================== PurePursuit算法的核心就是寻找前视追踪点, lookAheadDis需要自行调节(inche)====================*/
        std::pair<Point, bool> goal_pt_search(const std::vector<Point>& path, double lookAheadDis, int &lastFoundIndex) {
        
        // 提取当前X和当前Y
        double currentX = position->globalPoint.x;
        double currentY = position->globalPoint.y;

        // 初始化目标点, 以防没有点能够找到
        Point goalPt = path[0];
        bool intersectFound = false;
        // 从lastFoundIndex开始搜索即可，减小遍历次数
        int startingIndex = lastFoundIndex;
    
        for (int i = startingIndex; i < path.size() - 1; ++i) {
            // 开始线-圆交点计算代码
            double x1 = path[i].x - currentX;
            double y1 = path[i].y - currentY;
            double x2 = path[i+1].x - currentX;
            double y2 = path[i+1].y - currentY;
            double dx = x2 - x1;
            double dy = y2 - y1;
            double dr = std::sqrt(dx * dx + dy * dy);
            double D = x1 * y2 - x2 * y1;
            // ---------计算判别式----------
            double discriminant = (lookAheadDis * lookAheadDis) * (dr * dr) - D * D;
            //printf("discriminant : %lf", discriminant);
            // 判别式>=0,， 与直线有至少一个或两个交点
            if (discriminant >= 0) {
                // --------根据公式计算得到交点-------
                double sol_x1 = (D * dy + sgn(dy) * dx * std::sqrt(discriminant)) / (dr * dr);
                double sol_x2 = (D * dy - sgn(dy) * dx * std::sqrt(discriminant)) / (dr * dr);
                double sol_y1 = (-D * dx + std::abs(dy) * std::sqrt(discriminant)) / (dr * dr);
                double sol_y2 = (-D * dx - std::abs(dy) * std::sqrt(discriminant)) / (dr * dr);
                Point sol_pt1 = {sol_x1 + currentX, sol_y1 + currentY};
                Point sol_pt2 = {sol_x2 + currentX, sol_y2 + currentY};
                // --------判断交点是否在线段上------------
                double minX = std::min(path[i].x, path[i+1].x);
                double minY = std::min(path[i].y, path[i+1].y);
                double maxX = std::max(path[i].x, path[i+1].x);
                double maxY = std::max(path[i].y, path[i+1].y);

                // 与线段有一个或两个交点
                if ((minX <= sol_pt1.x && sol_pt1.x <= maxX && minY <= sol_pt1.y && sol_pt1.y <= maxY) || (minX <= sol_pt2.x && sol_pt2.x <= maxX && minY <= sol_pt2.y && sol_pt2.y <= maxY)) {
                    intersectFound = true;
                    
                    // 有两个交点
                    if ((minX <= sol_pt1.x && sol_pt1.x <= maxX && minY <= sol_pt1.y && sol_pt1.y <= maxY) && (minX <= sol_pt2.x && sol_pt2.x <= maxX && minY <= sol_pt2.y && sol_pt2.y <= maxY)) {
                        
                        if (pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1])) {
                            goalPt = sol_pt1;
                        } else {
                            goalPt = sol_pt2;
                        }
                    } else {
                        if (minX <= sol_pt1.x && sol_pt1.x <= maxX && minY <= sol_pt1.y && sol_pt1.y <= maxY) {
                            goalPt = sol_pt1;
                        } else {
                            goalPt = sol_pt2;
                        }
                    }
                   //printf("d1 : %lf , d2 : %lf \n", pt_to_pt_distance(goalPt, path[i+1]), pt_to_pt_distance({currentX, currentY}, path[i+1]));
                    // 只有当检测到目标点开始往回走了才会终止搜索;  否则只是更新一下lastFoundIndex，这样做可以保证之后的搜索不会往回走
                    if (pt_to_pt_distance(goalPt, path[i+1]) < pt_to_pt_distance({currentX, currentY}, path[i+1])) {
                        lastFoundIndex = i;
                        break;
                    } else {
                        lastFoundIndex = i + 1;
                    }

                }
            }
        }
      //  printf("intersectFound : %lf\n", intersectFound);
      //  printf("LastFoundIndex : %lf\n", lastFoundIndex);
        //  没有插入点, 返回path[i+1]
        if (!intersectFound) {
            goalPt = {path[lastFoundIndex].x, path[lastFoundIndex].y};
        }
        
        // 这里返回的目标点实际上没有angle的信息，需要根据需要在具体的底盘类的函数调用中去调整实现
        return {goalPt, intersectFound};
    }
            
};

}