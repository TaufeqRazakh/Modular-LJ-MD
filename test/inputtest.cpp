// Test the input parser with varaibales following C++ standard

#include <iostream> 
#include <fstream>        //std::ifstream
#include <sstream>
#include <vector>
#include <array>
#include <exception>      // std::exception, std::terminate

using namespace std;

int main(){
    ifstream ifs("pmd.in", ifstream::in);
    if(!ifs.is_open()) {
	cerr << "failed to open input file" << endl;
	terminate();
    }
    
    array<int,3> vproc;
    array<int,3> initUcell;
    double density, initTemp, deltaT;
    int stepLimit, stepAvg;
    
    ifs >> vproc[0] >> vproc[1] >> vproc[2];
    ifs >> initUcell[0] >> initUcell[1] >> initUcell[2];
    ifs >> density >> initTemp >> deltaT;
    ifs >> stepLimit >> stepAvg;

    cout << "Displaying values read" << endl;
    cout << vproc[0] << " " << vproc[1] << " " << vproc[2] << endl;
    cout << initUcell[0] << " " << initUcell[1] << " " << initUcell[2] << endl;
    cout << density << " " << initTemp << " " << deltaT << endl;
    cout << stepLimit  << " " << stepAvg << endl;
    
    cout << "Succcessfully finished test" << endl;
}
