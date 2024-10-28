#include "ConveyRing.h"
namespace tjulib 
{
    using namespace vex;

    int checkRing_() {
        int count = 0;

        printf("start checking\n");
        while (1) {
            while (1) {
                //printf("convey_belt.position:%lf\n", convey_belt.position(deg));
                //printf("vision continuing...\n");
                Vision.takeSnapshot(Blue3);
                if (Vision.objectCount > 0) {
                    count++;
                    abandon = true;
                    throwFlag = true;
                    //convey_belt.setPosition(0, deg);
                    printf("objectCount:%ld largestObject.height:%d\n", Vision.objectCount, Vision.largestObject.height);
                    //printf("other ring detected count:%d\n", count);
                    //printf("abandon:%d count:%d\n", abandon, count);
                }
                else {
                    count = 0;
                    //  abandon = false;
                }
                //task::sleep(100);
                wait(100, msec);
            }
            printf("vision end\n");
        }
        return 0;
    }


    // 检查环的颜色
    int CheckRing(){
        return 0;
    }
    // 将环扣上去
    int GetRing(){
        while(1){
            if(ring_convey_spin){
                
            }
        }
        return 0;
    }
    int getRing_() {

        bool reset = true;
        while (1) {
            //printf("start geting ring...\n");
            // printf("abandon:%d\n", abandon);
            static int count = 0;
      
            if (!abandon) {
                if (forwardSpin) {
                    roller_group.spin(forward, 100, pct);
                    convey_belt.spin(forward, 75, pct);
                }
                else if (reverseSpin) {
                    roller_group.spin(reverse, 100, pct);
                    convey_belt.spin(reverse, 75, pct);
                }
            }
            else {
                if (throwFlag == false)
                {
                    roller_group.stop();
                    convey_belt.stop();
                }
                else {
                    // count++;
                    // printf("throwing ring count:%d\n", count);
                    convey_belt.spin(forward, 100, pct);
                    wait(350, msec);
                    roller_group.stop();
                    convey_belt.stop();
                    //convey_belt.setVelocity(100, pct);
                    //convey_belt.spinFor(forward, 198.0, deg);
                    //abandon = false;
                    forwardSpin = false;
                    reverseSpin = false;
                    throwFlag = false;
                    reset = true;
                }
                //if (resetFlag == true) {

            //}
            //printf("abandon get ring\n");

            //abandon = false;
        }

    }
}
}
