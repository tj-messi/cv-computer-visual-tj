#pragma once
#include "tjulib-chassis/ordinary-chassis/ordi-straight.hpp"
#include "tjulib-chassis/ordinary-chassis/ordi-cur.hpp"

namespace tjulib
{
    using namespace vex;

    class Ordi_SmartChassis : public Ordi_StraChassis,  public Ordi_CurChassis{
    public:
        Ordi_SmartChassis(std::vector<std::vector<vex::motor*>*> &_chassisMotors, pidControl *_motorpidControl, Position*_position, const T _r_motor,
         pidControl *curpid, pidControl *fwdpid, pidControl *turnpid,T car_width)
            : Ordi_BaseChassis(_chassisMotors, _motorpidControl, _position, _r_motor), 
            Ordi_CurChassis(_chassisMotors, _motorpidControl, _position, _r_motor, curpid, car_width),
            Ordi_StraChassis(_chassisMotors, _motorpidControl, _position, _r_motor, fwdpid, turnpid)
        {};
    };
};