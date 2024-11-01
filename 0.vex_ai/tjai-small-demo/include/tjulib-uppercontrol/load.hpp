#pragma once
#include "vex.h"
class Load{
    #define MIDDLE 0
    #define LEFT -40
    #define RIGHT 45
    #define HALFLEFT -30
private:
    vex::motor* m;

public:
    Load(vex::motor* m) : m(m){
        // init position is RIGHT
        m->setPosition(RIGHT, deg);
        m->setBrake(vex::brakeType::hold);
        
    }
    void setPosition(double x){m->setPosition(x, deg);}
    void show(){printf("deg: %.2f\n", m->position(deg));}
    // V is always positive
    void kickLeft(double time, double V = 70){
        vexMotorVoltageSet(m->index(), V*120);
        task::sleep(time);
        vexMotorVoltageSet(m->index(), 0*120);
        m->stop(brake);
    }
    
    void kickRight(double time, double V = 70){
        vexMotorVoltageSet(m->index(), -V*120);
        task::sleep(time);
        vexMotorVoltageSet(m->index(), 0*120);
        m->stop(brake);
    }
    void holdForPush(double V = 70){
        m->spinTo(30, deg, V, vex::velocityUnits::pct);
        m->setBrake(vex::brakeType::hold);
        task::sleep(50);

    }
    void continuousKick(const int times, const int gapmTime = 200){
        for(int i=0;i<times;i++){  
            kickLeft(500, 180);  
            task::sleep(gapmTime);
            kickRight(500, 180);    
        }

    }
    // spin extra and hold
    void hold(){
        m->spin(vex::directionType::fwd, 100, vex::pct);
        task::sleep(300);
        m->stop(vex::brakeType::hold);
    }

};