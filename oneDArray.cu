//
// Created by brian on 11/5/2024.
//
#include <cassert>

#include "pla.h"
//~(numT)0 is max int

template<typename numT>
oneDArray<numT>::oneDArray(const oneDArray<numT> &other) {
    rows = other.rows;
    cols = other.cols;
    size = other.size;
    alrows = other.alrows;
    alcols = other.alcols;
    alsize = other.alsize;
    parent = other.parent;

    // Allocate memory for the array and copy the data
    arr = new numT[alsize];
    memcpy(arr, other.arr, other.alsize * sizeof(numT));
}
template oneDArray<uint8_t>::oneDArray(const oneDArray<uint8_t> &other);
template oneDArray<uint32_t>::oneDArray(const oneDArray<uint32_t> &other);

template <typename numT>
oneDArray<numT>::~oneDArray() {
    delete[] arr;
}
template oneDArray<uint8_t>::~oneDArray();
template oneDArray<uint32_t>::~oneDArray();

template <typename numT>
void oneDArray<numT>::initArr() {
    size = rows * cols;
    alsize = alrows * alcols;
    arr = new numT[alsize];
    memset(arr,0,alsize); //todo: maybe make this uint8_t instead of numT bc memset only works with bytes
}
template void oneDArray<uint8_t>::initArr();
template void oneDArray<uint32_t>::initArr();

template <typename numT>
NVCC_BOTH numT* oneDArray<numT>::ind_ptr(index_t row, index_t col) {
    return arr+ind(row,col);
}
template NVCC_BOTH uint8_t* oneDArray<uint8_t>::ind_ptr(index_t row, index_t col);
template NVCC_BOTH uint32_t* oneDArray<uint32_t>::ind_ptr(index_t row, index_t col);

// indexing function which returns 1d index
template <typename numT>
NVCC_BOTH index_t oneDArray<numT>::ind(index_t row, index_t col) const { //todo: make inline and move to .h file for speed, do profiling
    if (row >= alrows || col >= alcols) { //todo: maybe remove this check for speed
        //cerr << "arr index is out of allocated bounds-ind" << endl;
        //exit(1);
        OnGpuErr("arr index is out of allocated bounds--ind function");
    }
    return alcols * row + col;
}
template NVCC_BOTH index_t oneDArray<uint8_t>::ind(index_t row, index_t col) const;
template NVCC_BOTH index_t oneDArray<uint32_t>::ind(index_t row, index_t col) const;

template <>
NVCC_BOTH index_t oneDArray<uint32_t>::indOrdered(index_t row, index_t col) const { //for count array only
    if (row >= alrows || col >= alcols) {
        //cerr << "arr index is out of allocated bounds--indOrdered function" << endl;
        //exit(1);
        OnGpuErr("arr index is out of allocated bounds--indOrdered function");
    }
    if (row>col)
        return alcols * col + row;
    else
        return alcols * row + col;
}
template NVCC_BOTH index_t oneDArray<uint32_t>::indOrdered(index_t row, index_t col) const;

template <typename numT>
NVCC_BOTH index_pair oneDArray<numT>::indTwoD(index_t index) const {
    if (index>=alsize) { //todo: maybe remove this check for speed
        //cerr << "arr index is out of allocated bounds--indTwoD function" << endl;
        //exit(1);
        OnGpuErr("arr index is out of allocated bounds--indTwoD function");
    }
    uint32_t row = index/alcols; //int row = index/alcols;
    uint32_t col = index%alcols; //int col = index%alcols;
    if(row>=rows||col>=cols) {
        //cout<<"index is out of used bounds--indTwoD function"<<endl;
        OnGpuErr("index is out of used bounds--indTwoD function");
    }
    return {row,col};//index_pair(row,col);//make_pair(row,col);
}
template NVCC_BOTH index_pair oneDArray<uint8_t>::indTwoD(index_t index) const;
template NVCC_BOTH index_pair oneDArray<uint32_t>::indTwoD(index_t index) const;

// indexing function which returns value
template <typename numT>
NVCC_BOTH numT oneDArray<numT>::get_val(index_t row, index_t col) const { //todo: make inline and move to .h file for speed, do profiling
    if (row >= alrows || col >= alcols) { //todo: maybe remove this check for speed
        //cerr << "arr index is out of allocated bounds--get_val function" << endl;
        //exit(1);
        OnGpuErr("arr index is out of allocated bounds--get_val function");
    }
    return arr[alcols * row + col];
}
template NVCC_BOTH uint8_t oneDArray<uint8_t>::get_val(index_t row, index_t col) const;
template NVCC_BOTH uint32_t oneDArray<uint32_t>::get_val(index_t row, index_t col) const;

template <typename numT>
NVCC_BOTH void oneDArray<numT>::set_val(index_t row, index_t col, numT val) const {
    if (row >= alrows || col >= alcols) { //todo: maybe remove this check for speed
        //cerr << "arr index is out of allocated bounds, cannot set value at this index--set_val function" << endl;
        //exit(1);
        OnGpuErr("arr index is out of allocated bounds, cannot set value at this index--set_val function");
    }
    arr[alcols * row + col] = val;
}
template NVCC_BOTH void oneDArray<uint8_t>::set_val(index_t row, index_t col, uint8_t val) const;
template NVCC_BOTH void oneDArray<uint32_t>::set_val(index_t row, index_t col, uint32_t val) const;

template <>
void oneDArray<uint8_t>::printRow(index_t row) const {
    for (int i = 0; i < cols; i++)
        cout << char('0' + get_val(row, i));
}
template void oneDArray<uint8_t>::printRow(index_t row) const;

template <>
void oneDArray<uint32_t>::printRow(index_t row) const {
    for (int i = 0; i < cols; i++) {
        if(i <= row) {
            cout<<"-  ";
            continue;
        }
        uint32_t number = get_val(row, i);
        int count = floor(log10(number)) + 1;
        cout << number<< (count>1?" ":"  ");
    }
}
template void oneDArray<uint32_t>::printRow(index_t row) const;

template <>
void oneDArray<uint8_t>::printArr() const {
    for (index_t row = 0; row < alrows; row++) {
        if (!parent->usedrows[row])
            continue;
        printRow(row);
        cout << endl;
    }
}
template void oneDArray<uint8_t>::printArr() const;

template <>
void oneDArray<uint32_t>::printArr() const {
    for (index_t row = 0; row < rows; row++) {
        printRow(row);
        cout << endl;
    }
}
template void oneDArray<uint32_t>::printArr() const;

template <typename numT>
vector<string> oneDArray<numT>::convertBackFormat() {
    /*
    returns a vector of strings for each row, converted back into standard PLA format,
    re-adding the removed - and halfing the number of literals
    */
    vector<string> toReturn;
    for (index_t row = 0; row < alrows; row++) {
        if (!parent->usedrows[row])
            continue;
        string line;
        index_t start = ind(row, 0);
        for (index_t col = 0; col < cols; col=col+2) {
            if (arr[start + col] == 0 && arr[start + col + 1] == 0) {
                line.push_back('-');
            }
            else if (arr[start + col] == 1) {
                line.push_back('0');
            }
            else if (arr[start + col + 1] == 1) {
                line.push_back('1');
            }
        }
        toReturn.push_back(line);
    }
    return toReturn;
}
template vector<string> oneDArray<uint8_t>::convertBackFormat();
template vector<string> oneDArray<uint32_t>::convertBackFormat();

template <typename numT>
vector<string> oneDArray<numT>::convertToVecString() {
    //simply converts arr to a vector of strings with no additional modification
    vector<string> toReturn;
    for (index_t row = 0; row < alrows; row++) {
        if (!parent->usedrows[row])
            continue;
        string line;
        index_t start = ind(row, 0);
        for (index_t col = 0; col < cols; col++) {
            line.push_back('0' + arr[start + col]);
        }
        toReturn.push_back(line);
    }
    return toReturn;
}
template vector<string> oneDArray<uint8_t>::convertToVecString();
template vector<string> oneDArray<uint32_t>::convertToVecString();

template<> //todo: maybe change this to <> instead of <uint32_t>
index_t oneDArray<uint32_t>::findBiggestWeight(int minimumGain) const{
    //finds the biggest weight in the table

    index_t indMax,i;
    int temp,max;
    max = 0;
    indMax = 0;
    temp = 0;
    i = 0;
    cout<<"Finding the biggest weight..."<<endl;
    for (; i < alsize; i++) {
        temp = arr[i];
        if (temp >rows) //rows used to be passed in as itable.rows
            continue;
        if(temp>max) {
            max = temp;
            indMax = i;
        }
    }
    if((max < 2-minimumGain))
        return alsize;
    return indMax;
}
template index_t oneDArray<uint32_t>::findBiggestWeight(int minCost) const;