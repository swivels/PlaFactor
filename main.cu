#include <iostream>

#include "gpuutil.h"
#include "pla.h"

using namespace std;

int main(int argc, char **argv) { //set OMP_NUM_THREADS=16 bc 16 cores
    GpuInit(argc, argv);
    pla myPla;
    myPla.parsePla("./testcases/espresso_minimized.pla");
    myPla.fillCountArray();
    cout<<"Parsed tables in my formatted datastructure:"<<endl;
    myPla.printPla();
    cout<<"-----------------------------"<<endl<<endl;
    //myPla.printEspressoFormat();
    myPla.startMinimization();
    //myPla.printEspressoFormat();


    return 0;
}
