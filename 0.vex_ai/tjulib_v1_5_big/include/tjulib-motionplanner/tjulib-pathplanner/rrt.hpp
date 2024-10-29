#pragma once
#include "Math-Functions.h"
#include "tjulib-position/Position.hpp"
#include "tjulib-map/HighStakeMap.hpp"
#include <vector>
#include <random>

namespace tjulib {

    using namespace vex;
    typedef double T;

    struct Node {
        T x;
        T y;
        //T angle;
        int parent;
        
        // 默认构造函数
        Node() : x(0), y(0),  parent(-1) {}

        Node(double x, double y, double radius) : x(x), y(y), parent(-1) {}
    };

    

    class 
    RRT {
    public:
        RRT(const std::vector<Obstacle>& obstacleList,
            double minRand, double maxRand,
            double step = 2.0, int goalSampleRate = 10, int maxIter = 1000, T radius = 10)
            : obstacle_list(obstacleList), min_rand(minRand), max_rand(maxRand),
            step(step), goal_sample_rate(goalSampleRate),
            max_iter(maxIter), OMath(radius), radius(radius) {}

    

    std::vector<Point> rrt_planning_(Point start, Point goal) {
        // 初始化起始点和目标点
        this->start = Node(start.x, start.y, 0); // 起始点半径为0
        this->goal = Node(goal.x, goal.y, 0);     // 目标点半径为0
        node_list.push_back(this->start);// 加入起始点到节点列表中
        std::vector<Point> path = {};
        srand((unsigned int)(time(0)));
        for (int i = 0; i < max_iter; ++i) {
            Point rnd = sample();// 随机采样
            int n_ind = get_nearest_list_index(node_list, rnd);
            Node nearestNode = node_list[n_ind];

            double theta = std::atan2(rnd.y - nearestNode.y, rnd.x - nearestNode.x);
            Node newNode = get_new_node(theta, n_ind, nearestNode);

            if (check_segment_collision(newNode.x, newNode.y, nearestNode.x, nearestNode.y)) {//可以改成线和圆是否有交点
                node_list.push_back(newNode);
                if (is_near_goal(newNode)) {// 到达目标点
                    if (check_segment_collision(newNode.x, newNode.y, goal.x, goal.y)) {// 到达目标点后，判断是否有障碍物
                        int lastIndex = node_list.size() - 1;
                        path = get_final_course(lastIndex);
                        return path;
                    }
                }
            }

        }
        return path; // 如果没有找到路径，则返回空路径
    }
    
    // 将rrt_planning改为返回point版本的
    std::vector<Point> rrt_planning(Point start, Point goal){
        std::vector<Point>path_rev;
        path_rev = rrt_planning_({start.x, start.y, 0}, {goal.x, goal.y, 0});


        std::vector<Point>path;
        for(int i = path_rev.size()-2;i>=0;i--){
            path.push_back({path_rev[i].x, path_rev[i].y});
        }

        return path;
    }
    
    // 获取两线段之间的夹角(Deg 锐角)


    // 评估该规划的路径
    T estimate_path(std::vector<Point>path){
        std::vector<T>weight = {0.9, 0.025, 0.3};
        T score_length = 0;
        T score_turn = 0;
        T score_collision = 0;
        int n = path.size();

        for(int i = 1;i < n - 1;i++){
            // 计算路径长度，也是越小越好
            score_length += fabs(sqrt((path[i].x - path[i-1].x) * (path[i].x - path[i-1].x) + (path[i].y - path[i-1].y)* (path[i].y - path[i-1].y))); // 越小越好
            // 计算转角，越小越好
            score_turn += OMath.radiansToDegrees(OMath.angleBetweenLines(path[i - 1], path[i], path[i + 1])) / 180; 
            // 马上要碰撞了则记录一下
            for(auto obstacle : obstacle_list){
                 if(fabs(fabs(sqrt((path[i].x - obstacle.x) * (path[i].x - obstacle.x) + (path[i].y - obstacle.y)* (path[i].y - obstacle.y)))) <= 1.5 * (obstacle.radius + radius)){
                    score_collision += 1;
                }
            }

        }
        score_length += fabs(sqrt((path[n - 1].x - path[n - 2].x) * (path[n - 1].x - path[n - 2].x) + (path[n - 1].y - path[n - 2].y)* (path[n - 1].y - path[n - 2].y)));
        
        // 规范化
        score_length = (fabs(sqrt((path[0].x - path[n-1].x) * (path[0].x - path[n-1].x) + (path[0].y - path[n-1].y)* (path[0].y - path[n-1].y)))) / score_length;
        score_turn = 1 / (score_turn / n);
        score_collision = 1 / (score_collision / n);
        
       // printf(" score_length : %lf, score_turn : %lf, score_collision : %lf \n", score_length, score_turn, score_collision);
        return score_length * weight[0] + score_turn * weight[1] + score_collision * weight[2];
    }

    // 多次进行rrt规划，取其中最优路径
    std::vector<Point> optimal_rrt_planning(Point start, Point goal, int try_times = 2){
        std::vector<std::vector<Point>>Pathes;

        // 获取规划路径
        for(int i = 0; i<try_times;i++){
            Pathes.push_back(rrt_planning(start, goal));
            task::sleep(30);
        }

        // 对规划路径进行评估并计算其中最大值
        int optimal_no = -1;
        T max_score = INT_MIN;
        for(int i = 0; i< try_times;i++){
            //printf("no. %d path score: ", i + 1);
            T this_score;
            if((this_score = estimate_path(Pathes[i])) > max_score){
                max_score = this_score;
                optimal_no = i;
            }

        }
        
        return Pathes[optimal_no];
    }


    public:
        Node start;                                //起始点
        Node goal;                                 //目标点
        T min_rand = -72.0;                         //坐标系最小值
        T max_rand = 72.0;                          //坐标系最大值
        T step;                                     //步长
        T goal_sample_rate;                         //目标终点采样率
        T max_iter;                                 //最大迭代次数
        T radius;                                  //车子半径 
        Math OMath;
        std::vector<Obstacle> obstacle_list;        //障碍物列表
        std::vector<Node> node_list;

        Point sample() {
        Point rnd;
        if (rand() % 100 > goal_sample_rate) {
            rnd.x = rand_float(min_rand, max_rand);
            rnd.y = rand_float(min_rand, max_rand);
        } else {
            rnd.x = goal.x;
            rnd.y = goal.y;
        }
        return rnd;
    }

    Node get_new_node(double theta, int n_ind, Node& nearestNode) {
        Node newNode(nearestNode.x, nearestNode.y, 0); // 新节点半径为0
        newNode.x += step * std::cos(theta);
        newNode.y += step * std::sin(theta);
        newNode.parent = n_ind;
        return newNode;
    }

    bool is_near_goal(Node& node) {
        return line_cost(node, goal) < step;
    }

    static double line_cost(Node& node1, Node& node2) {
        return std::sqrt(std::pow(node1.x - node2.x, 2) + std::pow(node1.y - node2.y, 2));
    }

    // 找到与 rnd 最近的节点的索引
    int get_nearest_list_index(const std::vector<Node>& nodes, Point rnd) {
        double minDist = std::numeric_limits<double>::max();
        int minIndex = -1;

        for (size_t i = 0; i < nodes.size(); ++i) {
            double dist = std::pow(nodes[i].x - rnd.x, 2) + std::pow(nodes[i].y - rnd.y, 2);// 计算距离
            if (dist < minDist) {
                minDist = dist;
                minIndex = int(i);
            }
        }
        return minIndex;
    }

    double 
    rand_float(double a, double b) {
        return a + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (b - a)));
    }

    std::vector<Point> get_final_course(int lastIndex) {
        std::vector<Point> path;
        path.push_back(Point{goal.x, goal.y});
        while (node_list[lastIndex].parent != -1) {
            Node& node = node_list[lastIndex];
            path.push_back(Point{node.x, node.y});
            lastIndex = node.parent;
        }
        path.push_back(Point{start.x, start.y});
        return path;
    }

    bool check_segment_collision(double x1, double y1, double x2, double y2) {
        for (const auto& obs : obstacle_list) {
            double ox = obs.x;
            double oy = obs.y;
            double size = obs.radius + radius; // 使用障碍物的半径
            double dx = x2 - x1;
            double dy = y2 - y1;
            double a = (dy * (ox - x1) - dx * (oy - y1)) / sqrt(dx * dx + dy * dy);
            if (std::abs(a) < size) {
                return false; // 碰撞检测
            }
            T angle_pt = atan2(dy, dx);
                if((dy > 0 && dx > 0) || (dx < 0 && dy > 0))
                    angle_pt += 0;
                else if((dx > 0 && dy < 0) || (dx < 0 && dy < 0))
                    angle_pt += 360;
                else;
                angle_pt = 90 - angle_pt;
                if(angle_pt<0){
                    angle_pt+=360;
                }
               // if(std::fabs(GPS_.heading(deg) - angle_pt) >= 50){
                   // return false;
                //}
        }
        return true; // 没有碰撞
    }

      
    };
}