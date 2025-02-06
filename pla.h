//
// Created by Brian McElvain on 10/23/24.
//
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
#include "miscfunctions.h"
#include <algorithm> //for std::fill and all_of

using namespace std;

#define SECTION_START(name) // Start of section: name
#define SECTION_END(name)   // End of section: name

#ifndef PLACUDA_PLA_H
#define PLACUDA_PLA_H

class pla;

typedef uint32_t index_t;

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

    numT* ind_ptr(index_t row, index_t col);
    index_t ind(index_t row, index_t col) const;
    index_t indOrdered(index_t row, index_t col) const;

    pair<index_t,index_t> indTwoD(index_t index) const;
    numT get_val(index_t row, index_t col) const;
    void set_val(index_t row, index_t col,numT val) const;

    void printRow(index_t row) const;
    void printArr() const;

    vector<string> convertBackFormat(); //prints the pla table converted back to be in espresso format
    vector<string> convertToVecString(); //converts the pla table to a vector of strings

    index_t findBiggestWeight(int minCost) const;

    //GPU Functions
    void makeClones();

};

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

    int startCost = 0; //cost of the original pla table
    int iterationNumber = 1; //iteration number for the minimization process

    pla();
    //parsing functions
    void parsePla(string filename);
    void printPla();
    void printEspressoFormat(); //prints the pla table converted back to be in espresso format
    void printMaskArray();
    //array manipulation functions
    template <typename numT>
    void addColXtable(oneDArray<numT> & table);
    void addColItable();
    void addColOtable();
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
    index_t addLiteral(pair<index_t,index_t> bindex2d);

    int currentPlaCost();
    bool doesImprove(int oldCost, int expected);
    int itableSum();
    int otableSum();

    //GPU functions
    void makeClones();
};



#endif //PLACUDA_PLA_H
