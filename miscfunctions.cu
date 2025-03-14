#include "miscfunctions.h"

int nextPowTwo(int n) { //returns 2^x of which x is found by ceil(2^x = n)
    return pow(2, ceil(log2(n*2)));
}

vector<string> double_inames(vector<string> orig_ilist) {
    /*
    adds an assosiated ~literal for every literal in the list,
    this doubles the number of elements
    :param orig_list:
    :return: new_list
     */
    vector<string> new_list;
    for (auto i: orig_ilist) {
        new_list.push_back('~' + i);
        new_list.push_back(i);
    }
    return new_list;
}

vector<string> half_inames(vector<string> orig_ilist) {
    /*
    removes the assosiated ~literal for every literal in the list,
    this halfs the number of elements
     */
    vector<string> new_list;
    for(int i = 0; i < orig_ilist.size(); i+=2) {
        new_list.push_back(orig_ilist[i+1]);
    }
    return new_list;
}