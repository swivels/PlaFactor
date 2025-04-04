#include "gpuutil.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm> //for std::fill and all_of
#include "miscfunctions.h"

using namespace std;

//#define SECTION_START(name) // Start of section: name
//#define SECTION_END(name)   // End of section: name

#ifndef PLACUDA_PLA_H
#define PLACUDA_PLA_H

typedef uint32_t index_t;

enum arrayEnum{
    e_itable = 0, //0
    e_otable, //1
    e_countArray, //2
    e_usedrows, //3
    e_pairArray //4
};

struct index_pair{
    uint32_t first;
    uint32_t second;
    // Default constructor (for arrays)
    NVCC_BOTH index_pair() : first(0), second(0) {}
    // Parameterized constructor
    NVCC_BOTH index_pair(index_t f, index_t s) : first(f), second(s) {}
};

struct pair_weight {
    uint32_t pairIndex;
    uint32_t weight;
    pair_weight(): pairIndex(0), weight(0) {}
};

class pla;

template <typename numT>
struct oneDArray{
    //current number of used rows and columns
    index_t rows;
    index_t cols;
    index_t size;
    //allocated number of rows and columns
    index_t alrows; //also the number of overallocated rows in mask array (usedrows)
    index_t alcols;
    index_t alsize;

    numT *arr; //the main pla table stored in 1d array
    pla *parent;// stores a pointer to the parent pla for the struct


    oneDArray() : rows(0), cols(0), size(0), alrows(0), alcols(0), alsize(0), arr(nullptr), parent(nullptr) {}
    oneDArray(pla *parent) : rows(0), cols(0), size(0), alrows(0), alcols(0), alsize(0), arr(nullptr), parent(parent) {}
    oneDArray(const oneDArray<numT> &other);
    ~oneDArray();
    void initArr();

    NVCC_BOTH numT* ind_ptr(index_t row, index_t col);
    NVCC_BOTH index_t ind(index_t row, index_t col) const;
    NVCC_BOTH index_t indOrdered(index_t row, index_t col) const;

    NVCC_BOTH index_pair indTwoD(index_t index) const;
    NVCC_BOTH numT get_val(index_t row, index_t col) const;
    NVCC_BOTH void set_val(index_t row, index_t col,numT val) const;

    NVCC_BOTH void printRow(index_t row) const;
    NVCC_BOTH void printArr() const;

    vector<string> convertBackFormat(); //prints the pla table converted back to be in espresso format
    vector<string> convertToVecString(); //converts the pla table to a vector of strings

    NVCC_BOTH index_t findBiggestWeight(int minimumGain) const;

    //GPU Functions
    void retrieveDataGpu(oneDArray<numT> *Gpup);
};
//extern template struct oneDArray<uint8_t>;
//extern template struct oneDArray<uint32_t>;


class pla{
public:
    //original pla information
    string filename;
    int doti = 0; //# of original inputs
    int doto = 0; //# of original outputs
    vector <string> dotIlb; //list of original inputnames
    vector <string> dotOb; //list original outputnames
    int dotp = 0; //lines or number of product terms
    string dotType; //type of the file, "f" for f-type, "d" for d-type, etc
    //end original pla information

    vector <string> newIlb; //new input name list (~a, a, ~b, b, etc)
    vector <string> newOb; //new output name list
    oneDArray<uint8_t> itable; //input PLA table
    oneDArray<uint8_t> otable; //output PLA table
    uint8_t *usedrows; //array of used rows, 1 for used, 0 for unused
    oneDArray<uint32_t> countArray; //Count array representation of the pla file product term weights
    //<set> rowPairHash = nullptr #hash table for the pla table to store the product terms

    int pairCount = 0; //number of pairs in the pla table
    index_pair *pairArray; //array of pairs of row and column indexes for the pla table
    pair_weight *weightArray_d;

    int startCost = 0; //cost of the original pla table
    int iterationNumber = 1; //iteration number for the minimization process

    pla();
    //parsing functions
    void parsePla(string filename);
    void printPla();
    void printEspressoFormat(); //prints the pla table converted back to be in espresso format
    NVCC_BOTH void printMaskArray() const; //prints usedrows array
    //array manipulation functions
    template <typename numT>
    NVCC_BOTH void addColXtable(oneDArray<numT> & table);
    NVCC_BOTH void addColItable();
    NVCC_BOTH void addColOtable();
    void addColCountArray();
    int findFreeRowItable() const;
    index_t addRowsInOutTables(uint8_t* iline, uint8_t* oline); //adds 1 row to input and 1 row to output table, at the same row index
    void addRowCountArray(uint32_t *line);

    index_t resizeItable(int id);
    index_t resizeOtable(int id);
    index_t resizeXtable(oneDArray<uint8_t> &table, int id);
    index_t resizeCountArray(int id);

    void removeUnusedRows();
    void fillCountArray();
    //minimization functions
    void startMinimization();
    bool minimize();
    void startMinimizationGpu();
    bool minimizeGpu();
    index_t addLiteral(index_pair bindex2d);

    int currentPlaCost();
    bool doesImprove(int oldCost, int expected);
    int itableSum();
    int otableSum();

    //GPU functions
    void createPairArray();
    void makeClonesGpu();
    void launchPairArray();
    void retrieveDataGpu();
    void printDataOnGpu(arrayEnum type);
};

__global__ void print_Array_Kernel(pla* plaGpup,arrayEnum type);
__global__ void fill_CountArray_Kernel(pla* Gpup); //this is a kernel function<<<>>>
__global__ void findBigWCountArray_Kernel(pla* plaGpup);

#endif //PLACUDA_PLA_H
