#include <cassert>
#include "pla.h"
using namespace std;

const char* arrayEnumString[] = {"itable", "otable", "countArray","usedrows","pairArray"};

#define ParsingFunctions //code section for parsing functions
#ifdef ParsingFunctions

void parseLineToIntegerRow(string line, oneDArray<uint8_t> &table, bool isOutput, int linecount) { //initial conversion function when reading in txt file to array
    uint8_t *arrRow = new uint8_t[table.alcols];
    if(!isOutput) {
        for(index_t i = 0; i < line.size();i++) {
            index_t newInd = i*2;
            if(line[i] == '0') {
                arrRow[newInd] = 1;
                arrRow[newInd+1] = 0;
            }
            else if(line[i] == '1') {
                arrRow[newInd] = 0;
                arrRow[newInd+1] = 1;
            }else if(line[i] == '-'){
                arrRow[newInd] = 0;
                arrRow[newInd+1] = 0;
            }else {
                cerr<<"invalid input parsed from file, row cannot be added to input table array"<<endl;
                exit(1);
            }
        }
    }
    else {
        for(index_t i = 0; i < line.size();i++) {
            int t = line[i]-'0';
            if(t<0 || t>1)
                cerr<<"invalid input parsed from file, row cannot be added to output table array"<<endl;
            arrRow[i] = t;
        }
    }
    memcpy(table.arr + (linecount*table.alcols), arrRow, table.alcols*sizeof(uint8_t));
    delete []arrRow;
}

pla::pla() : itable(this), otable(this), countArray(this) {
}

void pla::parsePla(string filename) {
    this->filename = filename; // filename = espresso_minimized.pla

    cout<<"filename: "<<filename<<endl;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        return;
    }

    int lineno = 0;
    string line;

    getline(file,line);

    while(line[0] == '.')
    {
        vector <string> tokens;
        string tok;
        stringstream lineStream(line); //make line a stream
        while(getline(lineStream,tok,' ')) { //get tokens from line
            tokens.push_back(tok);
        }
        if(tokens[0] == ".i")
        {
            doti = stoi(tokens[1]);
            int tempdoti = 2*doti;
            itable.cols = tempdoti;
            int tempAlcols = nextPowTwo(tempdoti);
            itable.alcols = tempAlcols;
            countArray.cols = tempdoti;
            countArray.rows = tempdoti;
            countArray.alcols = tempAlcols;
            countArray.alrows = tempAlcols; // count array is a square matrix with the number of literals on both axis
        }
        else if(tokens[0] == ".o")
        {
            doto = stoi(tokens[1]);
            otable.cols = doto;
            otable.alcols = nextPowTwo(doto);
        }
        else if(tokens[0] == ".ilb")
        {
            for(int i = 1; i < tokens.size(); i++)
                dotIlb.push_back(tokens[i]);
        }
        else if(tokens[0] == ".ob")
        {
            for(int i = 1; i < tokens.size(); i++)
                dotOb.push_back(tokens[i]);
        }
        else if(tokens[0] == ".p")
        {
            dotp = stoi(tokens[1]);
            itable.rows = dotp;
            otable.rows = dotp;
            int alrows = nextPowTwo(dotp);
            itable.alrows = alrows;
            otable.alrows = alrows;
            usedrows = new uint8_t[alrows];
            fill_n(usedrows, dotp, 1);
            fill(usedrows + dotp, usedrows + alrows, 0);
        }
        else if(tokens[0] == ".type")
        {
            dotType = tokens[1];
        }
        getline(file,line); //get line from file
        lineno++;
    }
    newIlb = double_inames(dotIlb);
    newOb = dotOb;
    itable.initArr();
    otable.initArr();
    countArray.initArr();

    //content after line 5
    //parse the rest of the file
    cout<<"Initial parsing of pla table:"<<endl;
    cout<<"-----------------------------"<<endl<<endl;
    int linecount = 0;
    do{
        if(line[0] == '.' && line[1] == 'e')
            break;
        cout<<line<<endl;
        string iline,oline;
        stringstream lineStream(line); //make line a stream
        getline(lineStream,iline,' ');
        getline(lineStream,oline,'\n');
        parseLineToIntegerRow(iline, itable,false, linecount);
        parseLineToIntegerRow(oline, otable,true, linecount);
        linecount++;
    }while(getline(file,line));

    if (linecount != dotp) {
        cerr << "Error: number of product terms does not match the number of product terms specified in the header" << endl;
        exit(1);
    }
    cout<<endl;
    file.close();
}

void pla::printPla() {
    cout<<"itable: "<<endl;
    for ( int i = 0; i < itable.cols;i++) {
        if (i%10)
            cout<<" ";
        else
            cout<<(char)('0'+i/10);
    }
    cout<<endl;
    for ( int i = 0; i < itable.cols;i++)
        cout<<(char)('0'+i%10);
    cout<<endl;
    itable.printArr();
    cout<<endl;
    cout<<"otable: "<<endl;
    otable.printArr();
    cout<<endl;
    cout<<"countArray: "<<endl;
    countArray.printArr();
    cout<<endl;
}

void pla::printEspressoFormat() {//unfinished
    vector<string> reducedilb = half_inames(newIlb);
    cout<<endl;
    cout << ".i " << reducedilb.size() << endl;
    cout << ".o " << newOb.size() << endl;
    cout << ".ilb ";
    for (const auto& ilb : reducedilb)
        cout << ilb << " ";
    cout << endl;
    cout << ".ob ";
    for (const auto& ob : newOb)
        cout << ob << " ";
    cout << endl;
    cout << ".p " << dotp << endl;
    if (!dotType.empty())
        cout << ".type " << dotType << endl;

    vector<string> itableRows = itable.convertBackFormat();
    vector<string> otableRows = otable.convertToVecString();

    for (size_t i = 0; i < itableRows.size(); ++i) {
        cout << itableRows[i] << " " << otableRows[i] << endl;
    }

    cout << ".e" << endl;
}

NVCC_BOTH void pla::printMaskArray() const {
    printf("Mask array:\n");
    for(index_t row = 0; row < itable.alrows; row++) {
        printf("%c ", '0' +usedrows[row]);
    }
    printf("\n");
}

void pla::fillCountArray() {
    /*
    fill the count array with the number of overlapping literals in the itable using the input cubes
    calculated using a slower for loop method instead of the matrix multiplication method
    :return:
    print("Example small count array:")
    print("0(~a), 1(a), 2(b~), 3(b)")
    print("For example self.countArray[1,2]= 1 would fill the index of the 2d array where a and b~ are the literals of the product term")
    print("This means the product term reprentated by the row and column has 1 positive and 1 negative literal and occurs(weight) 1 time\n")
    print("Filling the count array:")
     */

    //cout<<"index result: ";
#pragma omp parallel for
    for(index_t col1 = 0; col1 < itable.cols; col1++) {
        for(index_t col2 = col1+1; col2 < itable.cols; col2++) {
            index_t resultIndex = countArray.ind(col1,col2);
            //cout<<resultIndex<<" with value: "; //todo: has some index's that are too large/out of bounds, fix
            int cnt = 0;
            for(index_t row = 0; row < itable.alrows; row++) {
                if(usedrows[row] == 0)
                    continue;
                if(itable.get_val(row,col1) == 0 || itable.get_val(row,col2) == 0)
                    continue;
                cnt++;
            }
            countArray.arr[resultIndex] = cnt;
            //cout<<countArray.arr[resultIndex]<<", "<<endl;
        }
    }
    //cout<<endl;

}

#endif ParsingFunctions

#define ArrayManipulationFunctions //code section for functions that manipulate itable, otable, and countArray arrays
#ifdef ArrayManipulationFunctions
template <>
void NVCC_BOTH pla::addColXtable<uint8_t>(oneDArray<uint8_t>& table) { // adds an empty column except for one index in it that stores data, which could be a 0 or 1
    if (table.cols == table.alcols) {
        resizeXtable(table,1); //2 options, one for itable and otable
    }

    for (index_t i = 0;i < table.alrows; i++) {
        table.arr[table.alcols*i+table.cols] = 0; //make the new column zero
    }
    table.cols++;
    table.size += table.rows; //size = rows * cols;
}

template <>
void NVCC_BOTH pla::addColXtable<uint32_t>(oneDArray<uint32_t>& table) { // adds an empty column except for one index in it that stores data, which could be a 0 or 1
    if (table.cols == table.alcols) {
        resizeCountArray(1);
    }

    for (index_t i = 0;i < table.alrows; i++) {
        table.arr[table.alcols*i+table.cols] = 0; //make the new column zero
    }
    table.cols++;
    table.size += table.rows; //size = rows * cols;
}

void NVCC_BOTH pla::addColItable(){addColXtable(itable);}
void NVCC_BOTH pla::addColOtable(){addColXtable(otable);}
void NVCC_BOTH pla::addColCountArray(){addColXtable(countArray);}

int pla::findFreeRowItable() const { // returns start row in 1d array for place to put new row in, works for Otable too
    if (itable.rows == itable.alrows)
        return -1;
    for (index_t i = 0; i < itable.alrows; i++) {
        if (usedrows[i] == 0) {
            usedrows[i] = 1;
            return i;
        }
    }
    return -1;
}

index_t pla::addRowsInOutTables(uint8_t* iline, uint8_t* oline) {
    /*Used to add a row to the input and output tables instead of adding them one by one using the addrows function because
     * the rowmask applies to both tables and the rowmask is updated in the addrows function, which would be called twice
     */
    int skipRow;
    int ifound = findFreeRowItable(); //returns row
    int ofound;
    if (ifound<0) {
        // this also means that rows == alrows
        skipRow = itable.alrows;
        ifound = resizeItable(0);
        ofound = resizeOtable(0); //returns loc
    }
    else { //else I found a free row and did not need to resize
        skipRow = ifound;
        ofound = otable.ind(ifound,0);
        ifound = itable.ind(ifound,0); //calcs loc
    }

    memcpy(itable.arr + ifound, iline, itable.cols*sizeof(uint8_t));
    memcpy(otable.arr + ofound, oline, otable.cols*sizeof(uint8_t));
    itable.rows++;
    itable.size+=itable.cols;
    otable.rows++;
    otable.size+=otable.cols;
    return skipRow; //returns row so that addliteral can return it so that minimize() can avoid substituing out the row just added to initiate the substitution
}                   //needed because rows are not contiguous/consecutive given the usedrows mask array

void pla::addRowCountArray(uint32_t *line) { //should only be called when adding a row to the count array--uint32_t version
    //int found = countArray.findFreeRow(); //returns row
    int found = -1;
    if (countArray.rows != countArray.alrows) //countArray Version of findFreeRow()
        found = countArray.rows;
    if (found<0) // this also means that rows == alrows
        found = resizeCountArray(0); //returns location
    else
        found = found * countArray.alcols; //calculates location if resize is not called

    memcpy(countArray.arr + found, line, countArray.cols*sizeof(uint32_t)); //will always add a line of all 0's so can just do it here as a row wont need custom initialization values
    countArray.rows++;
    countArray.size += countArray.cols; //size = rows * cols;
}

index_t pla::resizeItable(int id) {return resizeXtable(itable,id);}
index_t pla::resizeOtable(int id) {return resizeXtable(otable,id);}
//index_t pla::resizeCountArray(int id); //already below

//Resizing functions resize the array in the dimension specified by id to be the next power of 2 bigger
//this can be tricky because my 2d array is stored as a 1d array
//this function will also resize the rowmask array if resizing in the row dimension for itable and otable
//resizing columns is more complicated because it requires shifting all the data in the array
//resizeXtable is for both itable and otable resizing as they both use the same function
index_t pla::resizeXtable(oneDArray<uint8_t>& table, int id) { // if id = 0, then resize row, if id = 1 then resize column dimension
    int toReturn = 0;
    if (!id) // id = 0 to resize row
    { //todo: MAYBE DO REMOVE ROWS FUNCTION HERE
        cout<<"Resizing rows: "<<endl;
        index_t newrows = nextPowTwo(table.alrows);
        uint8_t *newarr = new uint8_t[newrows * table.alcols];
        memset(newarr,0,newrows * table.alcols);
        uint8_t *newmask = new uint8_t[newrows];
        memset(newmask,0,newrows);
        memcpy(newarr, table.arr, table.alsize*sizeof(uint8_t));
        memcpy(newmask, usedrows, table.alrows);
        delete[] table.arr;
        delete[] usedrows;
        table.arr = newarr;
        usedrows = newmask;
        table.alrows = newrows;
        toReturn = table.alsize; //todo: check if this is correct, the real answer might be alsize + cols
        table.alsize = table.alrows * table.alcols;
        return toReturn;
    } else // id = 1 to resize columns
    {
        cout<<"Resizing columns: "<<endl;
        if(table.cols != table.alcols)
            cout<<"error: do not need to resize columns, not at max"<<endl;//assert to make sure that all allocated columns are used up
        index_t newalcols = nextPowTwo(table.alcols);
        uint8_t *newarr = new uint8_t[table.alrows * newalcols];
        memset(newarr,0,(table.alrows * newalcols)*sizeof(uint8_t));
        for (index_t i = 0; i < table.alrows; i++) {
            uint8_t* dstptr = newarr + i * newalcols;
            memcpy(dstptr, table.arr + i * table.alcols, table.alcols*sizeof(uint8_t));
            //dstptr[alcols-1] = 0; //make the new column zero //todo now: check if this is correct, I think I should do this
        }
        delete[] table.arr;
        table.arr = newarr;
        table.alcols = newalcols;
        //toReturn = itable.alsize;//return doesnt matter for columns, not used
        table.alsize = table.alrows * table.alcols;
        return toReturn; //return doesnt matter for columns, not used
    }
}

index_t pla::resizeCountArray(int id) { // if id = 0, then resize row, if id = 1 then resize column dimension
    int toReturn = 0;
    if (!id) // id = 0 to resize row
    { //todo: MAYBE DO REMOVE ROWS FUNCTION HERE
        cout<<"resizing rows: "<<endl;
        index_t newrows = nextPowTwo(countArray.alrows);
        uint32_t *newarr = new uint32_t[newrows * countArray.alcols];
        memset(newarr,0,newrows * countArray.alcols);
        memcpy(newarr, countArray.arr, countArray.alsize*sizeof(uint32_t));
        delete[] countArray.arr;
        countArray.arr = newarr;
        countArray.alrows = newrows;
        toReturn = countArray.alsize; //todo: check if this is correct, the real answer might be alsize + cols
        countArray.alsize = countArray.alrows * countArray.alcols;
        return toReturn;
    } else // id = 1 to resize columns
    {
        cout<<"resizing columns: "<<endl;
        //assert(countArray.cols == countArray.alcols);//assert to make sure that all allocated columns are used up
        if(countArray.cols != countArray.alcols)
            cout<<"error: do not need to resize columns, not at max"<<endl;
        index_t newalcols = nextPowTwo(countArray.alcols);
        uint32_t *newarr = new uint32_t[countArray.alrows * newalcols];
        memset(newarr,0,(countArray.alrows * newalcols)*sizeof(uint32_t));
        for (index_t i = 0; i < countArray.alrows; i++) {
            uint32_t* dstptr = newarr + i * newalcols;
            memcpy(dstptr, countArray.arr + i * countArray.alcols, countArray.alcols*sizeof(uint32_t));
            //dstptr[alcols-1] = 0; //make the new column zero //todo now: check if this is correct, I think I should do this
        }
        delete[] countArray.arr;
        countArray.arr = newarr;
        countArray.alcols = newalcols;
        toReturn = countArray.alsize;
        countArray.alsize = countArray.alrows * countArray.alcols;
        return toReturn; //return doesnt matter for columns, not used
    }
}

void pla::removeUnusedRows() { //EXPAND REMOVE ROWS TO MERGE ROWS THAT HAVE ONLY 1 ONE IN THEM, MERGING THEIR OUTPUTS
    // removes any row (which is listed as used in the rowmask) from the table which is all zeros
    // it does this by checking the rowmask for the row, then checking the row in the table, if it is all zeros, then it removes the row
    // it removes the row by simply setting the rowmask to 0
    for (index_t i = 0; i < itable.alrows; i++) {
        if (!usedrows[i])
            continue;
        //add itable.cols outside of ind_ptr() to avoid error checking at the bounds due to the way all_of needs a last ptr past the end of a row
        if ((all_of(itable.ind_ptr(i,0), itable.ind_ptr(i,0)+itable.cols, [](uint8_t x) { return (x == 0); }))
            || (all_of(otable.ind_ptr(i,0), otable.ind_ptr(i,0)+otable.cols, [](uint8_t x) { return (x == 0); }))) {
            cout<<endl<<"I am inside removeUnusedRows and am removing row "<<i+1<<endl;
            usedrows[i] = 0;
            itable.rows--;
            otable.rows--;
        }
    }
}

#endif ArrayManipulationFunctions

#define MinimizingFunctions //code section for parsing functions
#ifdef MinimizingFunctions

void pla::startMinimization() {
    startCost = currentPlaCost();
    while(minimize())
        continue;
    return;
}

bool pla::minimize() {
    /*
    minimize the pla file by finding the highest weight product term as already calculated in the countarray and removing it by factoring/substitution
    and incrementally updating the count array and pla table

    Saving copies of the pla table, output table, and count array to revert to if the substitution does not improve the pla efficiency
    is too expensive, so instead the cost is calculated before substitution occurs and if the cost is higher after the substitution, a different weight is selected
     */
    printf("Minimizing pla table (iteration %d):\n",iterationNumber);
    int beforeCost = currentPlaCost();
    printf("Current PLA Cost %d\n",beforeCost);

    index_t bindex = countArray.findBiggestWeight(-1);
    if (bindex == countArray.alsize) {
        cout<<"Never found a cost efficient product term to substitute, aborting minimization."<<endl;
        return false;
    }
    index_t bweight = countArray.arr[bindex];
    index_pair bindex2d = countArray.indTwoD(bindex);
    index_t ind1 = bindex2d.first;
    index_t ind2 = bindex2d.second;
    printf("Biggest weight: %u\n",bweight);
    printf("Biggest weight index 1d: %u\n", bindex);
    printf("Biggest weight index 2d: (row %u, col %u)\n", ind1, ind2);

    //bweight*(-1*(number of literals being substituted out/saved) + 1(number of literals being added))+(number of literals being substituted out/saved);
    //bweight*(-2 + 1)+2;
    int sumCost = (int)bweight*-1 + 2; //cost of substitution
    cout<<"Cost of substitution: "<<sumCost<<endl;
    //cout<<"Cost of substitution is less than 0, proceeding with substitution"<<endl;
    index_t skipRow;
    bool onlyPair = false;
    for (index_t row = 0; row < itable.alrows; row++) {
        if (!usedrows[row])
            continue;
        //onlyPair = false; maybe can remove
        if (itable.get_val(row, ind1) == 1 && itable.get_val(row, ind2) == 1) {
            for (index_t col = 0; col < itable.cols; col++) {
                if (col == ind1 || col == ind2) //if neither of the substituted literals in the product term match this column
                    continue;
                if (itable.get_val(row, col) == 1) { //avoids using this row if it contains another literal other than the biggest weight substituted literals
                    onlyPair = false;
                    break;
                }
                onlyPair = true;
            }
            if(onlyPair) { //todo: collapse this section using the addLit function by sending a row number if this is true and -1 if false
                sumCost--; //todo now: move this code outside of the if statement, revisit code structure.
                printf("Found identical row to the product term being substituted, using this row, %u, instead of adding a new one\n", row);
                printf("New cost of substitution: %d\n", sumCost);

                addColItable();
                addColItable(); //1 col for ~a, 1 col for a
                newIlb.push_back(("~z_"+to_string(iterationNumber)));
                newIlb.push_back(("z_"+to_string(iterationNumber))); // uses same name as output being added because the output is being fed into and used as a literal in the input table

                addColOtable(); // new output column for the substituted product term to output to
                otable.set_val(row,otable.cols-1,1); //using the found product terms prexisting output row, and sets a new output to 1 while maintaining prexisting output contributions
                newOb.push_back(("z_"+to_string(iterationNumber)));

                addColCountArray();
                addColCountArray();
                uint32_t* newCountRow = new uint32_t[countArray.cols];
                memset(newCountRow, 0, countArray.cols * sizeof(uint32_t));
                addRowCountArray(newCountRow);
                addRowCountArray(newCountRow);
                delete[] newCountRow;

                skipRow = row;
                break;
            }

        }
    }
    if(!onlyPair)
        skipRow = addLiteral(bindex2d);
    //end new cost calc code
    //index_t skipRow = addLiteral(bindex2d);
    //cout<<"\nUpdating the pla table and count array now:"<<endl;
    //now to substitute the new product term into the pla table in place of product terms that contain it
    int testCount = 0;
    cout<<"Updating columns "<<ind1<<" and "<<ind2<<" in row: ";
#pragma omp parallel for
    for (index_t row = 0; row < itable.alrows; row++) { //launch kernel here
        if (!usedrows[row] || row == skipRow)
            continue;
        if (itable.get_val(row, ind1) == 1 && itable.get_val(row, ind2) == 1) {
            for (index_t col = 0; col < itable.cols-2; col++) {
                if (col == ind1 || col == ind2) //if neither of the substituted literals in the product term match this column
                    continue;
                if (itable.get_val(row, col) == 1) { //update count array, //use atomic decrement and increment
                    countArray.arr[countArray.indOrdered(col, ind1)]--; //by subtracting one from the previous combo between this column and one biggest index
                    countArray.arr[countArray.indOrdered(col, ind2)]--;
                    countArray.arr[countArray.ind(col, itable.cols-1)]++; //then add the combo between this column and the newly added literal
                }
            }
            cout<<row+1<<", ";
            testCount++;
            itable.set_val(row,ind1,0); //todo: can maybe test if its working better if i do -- instead of setting to 0
            itable.set_val(row,ind2,0);
            itable.set_val(row,itable.cols-1,1); //might be itable.cols-1
            countArray.arr[countArray.ind(ind1,ind2)]--; //subtract 1 from the highest weight product term location in count array for every substitution
        }
    }
    int testweight = countArray.arr[countArray.ind(ind1,ind2)];
    if(testweight != 0)
        cout<<"Error: count array was not updated correctly because bweight is "<<testweight<<", which is greater than 0. It should have been reduced to 0 here."<<endl;
    else
        countArray.arr[countArray.ind(ind1,ind2)]++; //add 1 back to the now substituted out highest weight product term location
    //in count array to account for the single added row which contained the substituted out product term
    cout<<endl<<testCount<<" rows updated."<<endl;
    removeUnusedRows(); //EXPAND REMOVE ROWS TO MERGE ROWS THAT HAVE ONLY 1 ONE IN THEM, MERGING THEIR OUTPUTS

    cout<<endl<<endl<<"Minimized table results:"<<endl;
    printPla();
    if(doesImprove(beforeCost,sumCost)) {
        cout<<"Substitution improved the pla efficiency"<<endl;
        cout<<"Iteration number "<<iterationNumber++<<" successful."<<endl;
        cout<<"-----------------------------"<<endl<<endl;
        return true;
    }else {
        cout<<"Error: Substitution did not improve the pla efficiency"<<endl;
        cout<<"Iteration number "<<iterationNumber<<" failed."<<endl;
        cout<<"Started with a cost of "<<startCost<<" and "<<doti<<" literals. Ended with a cost of "<<currentPlaCost()<<" and "<<itable.cols/2<<" literals."<<endl;
        cout<<"-----------------------------"<<endl<<endl;
        return false;
    }
}

index_t pla::addLiteral(index_pair bindex2d) {
    index_t ind1 = bindex2d.first;
    index_t ind2 = bindex2d.second;
    //cout<<"Biggest weight index: "<<ind1<<", "<<ind2<<endl;

    addColItable();
    addColItable(); //1 col for ~a, 1 col for a
    newIlb.push_back(("~z_"+to_string(iterationNumber)));
    newIlb.push_back(("z_"+to_string(iterationNumber)));

    addColOtable(); // new output column for the substituted product term to output to
    newOb.push_back(("z_"+to_string(iterationNumber)));

    uint8_t* newInpArrRow = new uint8_t[itable.cols]; //new input row to add to the itable
    fill(newInpArrRow,newInpArrRow+itable.cols,0);
    newInpArrRow[ind1] = 1;
    newInpArrRow[ind2] = 1;

    uint8_t* newOutArrRow = new uint8_t[otable.cols]; //new output row to add to the otable
    fill(newOutArrRow,newOutArrRow+otable.cols,0);
    newOutArrRow[otable.cols-1] = 1;

    index_t skipRow = addRowsInOutTables(newInpArrRow,newOutArrRow);
    delete []newInpArrRow;
    delete []newOutArrRow;

    addColCountArray();
    addColCountArray();
    uint32_t* newCountRow = new uint32_t[countArray.cols];
    //fill(newCountRow,newCountRow+countArray.cols,0); //todo: replace fill with memset so its easily transferable to the gpu
    memset(newCountRow, 0, countArray.cols * sizeof(uint32_t));
    addRowCountArray(newCountRow);
    addRowCountArray(newCountRow);
    delete[] newCountRow;

    return skipRow;
}

int pla::currentPlaCost() {
    return itableSum()+otableSum();
}

bool pla::doesImprove(int oldCost, int expected) { //count the number of inputs to gates
    /*
    *sum up pla table except for "product terms" with a single literal,
     sum up output table except for columns with a single output/1 contribution
     then add those together and compare the sum of the original pla table and output table
     */
    int newCost = currentPlaCost();

    int actual = newCost-oldCost;
    cout<<"Old Cost: "<<oldCost<<", New Cost: "<<newCost<<endl<<"Expected difference: "<<expected<<", Found Difference: "<<actual<<endl;
    if (actual != expected)
        cout<<"Error: Expected difference in cost does not match actual difference"<<endl;
    if (newCost >= oldCost)
        return false;
    else
        return true;
}

int pla::itableSum() {
    int sum = 0;
    int rowsum;
    for (int row = 0; row < itable.rows;row++) {
        if(!usedrows[row])
            continue;
        rowsum = 0;
        for(int col = 0;col<itable.cols;col++) {
            rowsum += (int)itable.get_val(row,col);
        }
        if(rowsum == 1) {
            //cout<<"test " <<rowsum<<endl;
            continue; // todo: make sure this is correct, copied from python code
        }
        //cout<<rowsum<<", ";
        sum += rowsum;
    }
    //cout<<endl;
    //cout<<sum<<endl;
    return sum;
}

int pla::otableSum() {
    int sum = 0;
    int colsum;
    for(int col = 0;col<otable.cols;col++) {
        colsum = 0;
        for (int row = 0; row < otable.rows;row++) {
            if(!usedrows[row]) //todo: maybe can zero all unused rows and then sum up the columns without needing to check used rows as val can only be 1 or 0
                continue;
            colsum += otable.get_val(row,col);
        }
        if(colsum == 1)
            continue;
        sum+=colsum;
    }
    return sum;
}

#endif MinimizingFunctions

#define GPUFunctions //code section for functions for GPU
#ifdef GPUFunctions

void pla::createPairArray() {
    pairCount = itable.cols*(itable.cols-1)/2;
    pairArray = new index_pair[pairCount];
    assert(pairCount == ((countArray.size-countArray.rows)/2));
    index_t pos = 0;
    for(index_t col1 = 0; col1 < itable.cols; col1++) {
        for(index_t col2 = col1+1; col2 < itable.cols; col2++) {
            pairArray[pos++] = index_pair(col1,col2);
        }
    }
#if 0
    for (int i = 0; i < pairCount; i++) {
        cout<<"("<<pairArray[i].first<<", "<<pairArray[i].second<<"), ";
        if (i%5 == 0)
            cout<<endl;
    }
#endif
}

__global__ void patchPlaParentPointer(pla* plap) {
    plap->itable.parent = plap;
    plap->otable.parent = plap;
    plap->countArray.parent = plap;
}

void pla::makeClonesGpu() {
    cout<<"Cloning Pla and subtables to GPU"<<endl;
    GPUDECLPATCHES(5);
    ADDPATCH(this, itable.arr);
    ADDPATCH(this, otable.arr);
    ADDPATCH(this, countArray.arr);
    ADDPATCH(this, usedrows);
    ADDPATCH(this, pairArray);
    ENDPATCH();
    GpuCloneForDevices(itable.arr,itable.alsize*sizeof(uint8_t),true);
    GpuCloneForDevices(otable.arr,otable.alsize*sizeof(uint8_t),true);
    GpuCloneForDevices(countArray.arr,countArray.alsize*sizeof(uint32_t),false);
    GpuCloneForDevices(usedrows,itable.alsize*sizeof(uint8_t),true);
    GpuCloneForDevices(pairArray,pairCount*sizeof(index_pair),true);

    GpuCloneForDevices(this,sizeof(*this),true,_patches);

    pla* plaGpup = (pla*)GpuFindCloneDevice((void*)this,GpuGetDevice());
    patchPlaParentPointer<<<1,1>>>(plaGpup);

    GpuCheckKernelLaunch("patchPlaParentPointer");
}

void pla::launchPairArray() {
    int threads = pairCount;
    int dev = GpuGetDevice();//get device for current thread on CPU--will be thread 0. gets dev which is gpu for thread 0
    pla *plaGpup = (pla*)GpuFindCloneDevice((void*)this,dev);
    int bcnt = GpuThreads2BlockCount(threads);
    fill_CountArray_Kernel<<<bcnt,GpuBlockSize>>>(plaGpup); //GpuBlockSize is global and set by GPU init

    GpuCheckKernelLaunch("fill_CountArray_Kernel");
}

void pla::retrieveDataGpu(){
    int dev = GpuGetDevice();
    void *LocationGpu = GpuFindCloneDevice((void*)this,dev);
    char* buf = new char[sizeof(pla)];
    GpuMemcpyDeviceToHost(buf,LocationGpu, sizeof(pla));
    pla* plaGpu = (pla*)buf;

    countArray.retrieveDataGpu(&plaGpu->countArray);
    itable.retrieveDataGpu(&plaGpu->itable);
    otable.retrieveDataGpu(&plaGpu->otable);

    //countArray.printArr();
    delete[] buf;
}

void pla::printDataOnGpu(arrayEnum type) {
    printf("Printing %s on GPU\n",arrayEnumString[type]);
    void *LocationGpu = GpuFindCloneDevice((void*)this,GpuGetDevice());
    print_Array_Kernel<<<1,1>>>((pla*)LocationGpu,type);
    GpuCheckKernelLaunch("print_Array_Kernel");
}

__global__ void print_Array_Kernel(pla* plaGpup,arrayEnum type) {
    switch (type) {
        case e_itable:
            plaGpup->itable.printArr();
        break;
        case e_otable:
            plaGpup->otable.printArr();
        break;
        case e_countArray:
            plaGpup->countArray.printArr();
        break;
        case e_usedrows:
            plaGpup->printMaskArray();
        break;/*
        case e_pairArray:
            for (int i = 0; i < plaGpup->pairCount; i++) {
                printf("(%u, %u), ", plaGpup->pairArray[i].first, plaGpup->pairArray[i].second);
                if (i % 5 == 0)
                    printf("\n");
            }
        break;*/
        default:
            printf("Unknown array type in print_Array_Kernel\n");
        break;
    }
}

__global__ void fill_CountArray_Kernel(pla* plaGpup){ //__global__ means it lives on the gpu and can be called from the cpu
    int idx = threadIdx.x+blockDim.x*blockIdx.x; //way to find global thread index within a device
	// idx is the "sequence" number
    if (idx >= plaGpup->pairCount)
        return;
    index_t col1 = plaGpup->pairArray[idx].first;
    index_t col2 = plaGpup->pairArray[idx].second;
    //printf("Thread %d: col1: %u, col2: %u\n",idx,col1,col2);

	index_t resultIndex = plaGpup->countArray.ind(col1,col2); //result index is unique to the thread
    int cnt = 0;
    for(index_t row = 0; row < plaGpup->itable.alrows; row++) {
        if(plaGpup->usedrows[row] == 0)
            continue;
        if(plaGpup->itable.get_val(row,col1) == 0 || plaGpup->itable.get_val(row,col2) == 0)
            continue;
        cnt++; //counts with a thread local variable if pair of literals exist in any rows
    }
    plaGpup->countArray.arr[resultIndex] = cnt;
}


/*
*#define NVCC_BOTH __host__ __device__
// This means only on the GPU
#define NVCC_DEVICE __device__ //__global__ means it lives on the gpu and can be called from the cpu
#endif
 *
 *
 */


#endif GPUFunctions