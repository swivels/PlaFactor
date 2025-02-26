#include <iostream>
#include "argparse.hpp"
#include "pla.h"

using namespace std;

int main(int argc, char **argv) {
    //set OMP_NUM_THREADS=16 bc 16 cores
    //https://github.com/p-ranav/argparse?tab=readme-ov-file#quick-start
    argparse::ArgumentParser program("PlaCuda");
    program.add_argument("--gpu")
      .help("use GPU? 1 for yes, 0 for no")
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
    }
    else {
        std::cout << "Gpu disabled" << std::endl;
        std::cout << "Running on CPU only" << std::endl;
    }
    GpuInit(argc, argv);
    pla myPla;
    myPla.parsePla("./testcases/espresso_minimized.pla");
    myPla.fillCountArray();
    cout<<"Parsed tables in my formatted datastructure:"<<endl;
    myPla.printPla();
    cout<<"-----------------------------"<<endl<<endl;
    //myPla.printEspressoFormat();
    myPla.createPairArray();
    myPla.makeClonesGpu();
    myPla.launchPairArray(); //fills in countArray on GPU
    myPla.startMinimization();
    //myPla.printEspressoFormat();



    return 0;
}
