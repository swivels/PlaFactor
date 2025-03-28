#include <iostream>
#include "argparse.hpp"
#include "pla.h"
//PLA is a two level logic circuit and my program turns it into an efficient,
//minimized multilevel logic circuit
//test comment
using namespace std;

int main(int argc, char **argv) {
    //https://github.com/p-ranav/argparse?tab=readme-ov-file#quick-start
    argparse::ArgumentParser program("PlaCuda");
    program.add_argument("--gpu")
      .help("use GPU? --gpu for yes, --nogpu for no")
      .default_value(false)
      .implicit_value(true);
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    if (program["--gpu"] == true) {
        std::cout << "Gpu enabled" << std::endl;
        GpuInit(argc, argv);
        pla myPlaGpu;
        myPlaGpu.parsePla("./testcases/espresso_minimized.pla");
        //myPlaGpu.fillCountArray();
        cout<<"Parsed tables in my formatted datastructure:"<<endl;
        myPlaGpu.printPla();
        cout<<"-----------------------------"<<endl<<endl;
        myPlaGpu.createPairArray();
        myPlaGpu.makeClonesGpu();
        myPlaGpu.launchPairArray(); //fills in countArray on GPU
        //myPlaGpu.retrieveDataGpu();
        myPlaGpu.startMinimizationGpu();
        //myPlaGpu.printDataOnGpu(e_countArray);
    }
    else {
        std::cout << "Gpu disabled" << std::endl;
        std::cout << "Running on CPU only" << std::endl;
        pla myPla;
        myPla.parsePla("./testcases/espresso_minimized.pla");
        myPla.createPairArray();
        myPla.fillCountArray();
        cout<<"Parsed tables in my formatted datastructure:"<<endl;
        myPla.printPla();
        cout<<"-----------------------------"<<endl<<endl;
        myPla.startMinimization();
        //myPla.printEspressoFormat();
    }
    //myPla.printEspressoFormat();



    return 0;
}
