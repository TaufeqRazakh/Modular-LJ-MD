#include <iostream>
#include <cmath>
#include <fstream>      // std::fstream
#include <vector>
#include <array>
#include <algorithm>
#include <string>
#include <sstream>
#include <map>
#include "mpi.h"

/* Constants for the random number generator */
#define D2P31M 2147483647.0
#define DMUL 16807.0

using namespace std;

class Atom{
  public:
  double type;          // identifier for atom type
  bool isResident;
  
  double x;		// position in x axis
  double y;		// position in y axis
  double z;		// position in y axis

  double ax;		// acceleration on x axis
  double ay;		// acceleration on y axis
  double az;		// acceleration on y axis
  
  double vx;		// velocity on x axis
  double vy;		// velocity on y axis
  double vz;            // velocity on y axis
  // Default constructor
  Atom(); 
};

class SubSystem{
  public:
  int pid; //sequential processor ID of this cell
  int n; // Number of resident atoms in this processor
  int nglob; // Total number of atoms summed over processors
  double comt; // elapsed wall clock time & Communication time in second
  array<double, 3> al; // Box length per processor
  array<int, 3> vid; /* Vector index of this processor */
  array<int, 3> myparity; // Parity of this processor
  array<int, 6> nn; // Neighbor node list of this processor
  vector<vector<double> > sv; // Shift vector to the 6 neighbors
  array<double, 3> vSum, gvSum;
  vector<Atom> atoms;

  double kinEnergy,potEnergy,totEnergy,temperature;
  
  /* Create subsystem with parameters input parameters to calculate 
     the number of atoms and give them random velocities */
  SubSystem(int sid, array<int, 3> vproc, array<int, 3> InitUcell, double InitTemp, double Density);

  void InitNeighborNode(array<int, 3> vproc);

   // Update Atomic co-ordinates to r(t+Dt)
  void Update(double DeltaT);

  // Update the velocities after a time-step DeltaT
  void Kick(double DeltaT);

  // Exchange boundaty-atom co-ordinates among neighbor nodes
  void AtomCopy();

  // Send moved-out atoms to neighbor nodes and receive moved-in atoms
  // from neighbor nodes
  void AtomMove();

  // Return true if an Atom lies in them boundary to a neighbor ID
  int bbd(Atom atom, int ku);

  // Return true if an Atom lies in them boundary to a neighbor ID
  int bmv(Atom atom, int ku);

  //Evaluates physical properties: kinetic, potential & total energies
  void EvalProps(int stepCount, double DeltaT);

  // Write out XYZ co-ordinates from each frame into separate files
  void WriteXYZ(int step);

  static double Dmod(double a, double b) {
    int n;
    n = (int) (a / b);
    return (a - b * n);
  }
  static double RandR(double *seed) {
    *seed = Dmod(*seed*DMUL,D2P31M);
    return (*seed/D2P31M);
  }
  static void RandVec3(double *p, double *seed) {
    double x = 0, y = 0,s = 2.0;
    while (s > 1.0) {
      x = 2.0*RandR(seed) - 1.0; y = 2.0*RandR(seed) - 1.0; s = x*x + y*y;
    }
    p[2] = 1.0 - 2.0*s; s = 2.0*sqrt(1.0 - s); p[0] = s*x; p[1] = s*y;
  }  
};

void SingleStep(SubSystem&, double);
void ComputeAccel(SubSystem&);
