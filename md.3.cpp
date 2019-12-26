#include <numeric>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <sys/time.h>
#include <sstream>
#include <string>
#include <fstream>      // std::fstream
#include "mpi.h"
#undef SEEK_SET
#undef SEEK_CUR
#undef SEEK_END

const double ARmass = 39.94800000; //A.U.s
const double ARsigma = 3.40500000; // Angstroms
const double AReps   = 119.800000; // Kelvins
const double CellDim = 12.0000000; // Angstroms
int NPartPerCell = 10;

using namespace std;
// Class for keeping track of the properties for a particle
class Particle{
public:
  double x;		// position in x axis
  double y;		// position in y axis
  double z;		// position in y axis
  double fx;		// total forces on x axis
  double fy;		// total forces on y axis
  double fz;		// total forces on y axis
  double ax;		// acceleration on x axis
  double ay;		// acceleration on y axis
  double az;		// acceleration on y axis
  
  double vx;		// velocity on x axis
  double vy;		// velocity on y axis
  double vz;            // velocity on y axis
  // Default constructor
  Particle() 
    : x(0.0),y(0.0),z(0.0),fx(0.0),fy(0.0),fz(0.0),ax(0.0),
      ay(0.0),az(0.0),vx(0.0),vy(0.0),vz(0.0){
  }
  
  double update(){
    // We are using a 1.0 fs timestep, this is converted
    double DT = 0.000911633;
    ax = fx/ARmass;
    ay = fy/ARmass;
    az = fz/ARmass;
    vx += ax*0.5*DT;
    vy += ay*0.5*DT;
    vz += ay*0.5*DT;
    x += DT*vx;
    y += DT*vy;
    z += DT*vz;
    fx = 0.0;
    fy = 0.0;
    fz = 0.0;
    return 0.5*ARmass*(vx*vx+vy*vy+vz*vz);
  }
};
 

class Cell {
public:
  vector<Particle> particles;
  // Buffer to hold coordinates recieved
  vector< vector< double > > remote_particles;
  // Buffer to hold coordinates to send
  vector<double> PartCoordsBuff;
  int absx,absy,absz;
  MPI_Request reqr[8],reqs[8];
  int nreqr, nreqs;
  Cell(int x, int y, int z, int ax, int ay, int az, int nParticles) : particles(nParticles){
    absx = ax;
    absy = ay;
    absz = az;
    //  We will be filling the cells, making sure than
    //  No atom is less than 2 Angstroms from another
    double rcelldim = double(CellDim);
    double centerx = rcelldim*double(ax) + 0.5*rcelldim;
    double centery = rcelldim*double(ay) + 0.5*rcelldim;
    double centerz = rcelldim*double(az) + 0.5*rcelldim;
   
    // Randomly initialize particles to be some distance 
    // from the center of the square
    // place first atom in cell
    particles[0].x = centerx + ((drand48()-0.5)*(CellDim-2));
    particles[0].y = centery + ((drand48()-0.5)*(CellDim-2));
    particles[0].z = centerz + ((drand48()-0.5)*(CellDim-2));
    
    double r,rx,ry;
    for (int i = 1; i < particles.size(); i++) {
      r = 0;
      while(r < 2.7) {   // square of 2
	r = 2.70001;
	rx = centerx + ((drand48()-0.5)*(CellDim-2)); 
	ry = centery + ((drand48()-0.5)*(CellDim-2));
      	//loop over all other current atoms
	for(int j = 0; j<i; j++){
	  double rxj = rx - particles[j].x;
	  double ryj = ry - particles[j].y;
	  double rzj = ry - particles[j].z;
	  double rt = rxj*rxj+ryj*ryj+rzj*rzj;
	  if(rt < r) r=rt;
	}
      }
      particles[i].x = rx;
      particles[i].y = ry;
      particles[i].z = rz;
    }
  }
  
  void Communicate(int*** CellMap, int Dimension){
    // Pack the message
    PartCoordsBuff[0] = particles.size();
    for(int i=0;i<particles.size();i++){
      PartCoordsBuff[1+i*2] = particles[i].x;
      PartCoordsBuff[1+i*2+1] = particles[i].y;
    }
    
    // Assuming a good MPI layer, communication between two of the same 
    // processor should be pretty efficient.  Not screening greatly simplifies
    // the code
    
    // i+1,j
    int ircount =0;
    if(absx!=Dimension-1){
      int tag = (absx+1)*Dimension+absy;
      MPI_Irecv(remote_particles[ircount],NPartPerCell*2+1,
      		MPI_DOUBLE,CellMap[absx+1][absy][0],
      		tag,MPI_COMM_WORLD,&reqr[ircount]);
      ircount++;
    }
    //i-1,j
    if(absx!= 0){
      int tag = (absx-1)*Dimension+absy;
      MPI_Irecv(remote_particles[ircount],NPartPerCell*2+1,
		MPI_DOUBLE,CellMap[absx-1][absy][0],
		tag,MPI_COMM_WORLD,&reqr[ircount]);
      ircount++;
    }
    // i,j+1
    if(absy!=Dimension-1){
      int tag = (absx)*Dimension+(absy+1);
      MPI_Irecv(remote_particles[ircount],NPartPerCell*2+1,
		MPI_DOUBLE,CellMap[absx][absy+1][0],
		tag,MPI_COMM_WORLD,&reqr[ircount]);
      ircount++;
    }
    // i,j-1
    if(absy !=0){
      int tag = absx*Dimension+(absy-1);
      MPI_Irecv(remote_particles[ircount],NPartPerCell*2+1,
		MPI_DOUBLE,CellMap[absx][absy-1][0],
		tag,MPI_COMM_WORLD,&reqr[ircount]);
      ircount++;
    }
    // i+1,j+1
    if(absx != Dimension-1 && absy != Dimension-1){
      int tag = (absx+1)*Dimension + (absy+1);
      MPI_Irecv(remote_particles[ircount],NPartPerCell*2+1,
		MPI_DOUBLE,CellMap[absx+1][absy+1][0],
		tag,MPI_COMM_WORLD,&reqr[ircount]);
      ircount++;
    }
    //i+1,j-1
    if(absx != Dimension-1 && absy != 0){ 
      int tag = (absx+1)*Dimension + (absy-1);
      MPI_Irecv(remote_particles[ircount],NPartPerCell*2+1,
		MPI_DOUBLE,CellMap[absx+1][absy-1][0],
		tag,MPI_COMM_WORLD,&reqr[ircount]);
      ircount++;
    }
    //i-1,j+1
    if(absx != 0 && absy != Dimension-1){
      int tag = (absx-1)*Dimension + (absy+1);
      MPI_Irecv(remote_particles[ircount],NPartPerCell*2+1,
		MPI_DOUBLE,CellMap[absx-1][absy+1][0],
		tag,MPI_COMM_WORLD,&reqr[ircount]);
      ircount++;
    }
    //i-1,j-1
    if(absx !=0 && absy !=0){
      int tag = (absx-1)*Dimension + (absy-1);
      MPI_Irecv(remote_particles[ircount],NPartPerCell*2+1,
		MPI_DOUBLE,CellMap[absx-1][absy-1][0],
	       	tag,MPI_COMM_WORLD,&reqr[ircount]);
      ircount++;
    }
    
    // Now the sending
    int iscount = 0;

    //i+1,j   
    if(absx != Dimension-1){
      int tag = absx*Dimension + absy;
      MPI_Isend(PartCoordsBuff,NPartPerCell*2+1,MPI_DOUBLE,CellMap[absx+1][absy][0],
		tag,MPI_COMM_WORLD,&reqs[iscount]);
      iscount++;
    }
    // i-1,j
    if(absx != 0){
      int tag = absx*Dimension + absy;
      MPI_Isend(PartCoordsBuff,NPartPerCell*2+1,MPI_DOUBLE,CellMap[absx-1][absy][0],
		tag,MPI_COMM_WORLD,&reqs[iscount]);
      iscount++;
    }
    // i,j+1
    if(absy != Dimension-1){
      int tag = (absx)*Dimension + (absy);
      MPI_Isend(PartCoordsBuff,NPartPerCell*2+1,MPI_DOUBLE,CellMap[absx][absy+1][0],
		tag,MPI_COMM_WORLD,&reqs[iscount]);
      iscount++;
    }
    // i,j-1
    if(absy != 0){
      int tag = (absx)*Dimension + (absy);
      MPI_Isend(PartCoordsBuff,NPartPerCell*2+1,MPI_DOUBLE,CellMap[absx][absy-1][0],
		tag,MPI_COMM_WORLD,&reqs[iscount]);
      iscount++;
    }
    // i+1,j+1
    if(absx != Dimension-1 && absy != Dimension-1){
      int tag = (absx)*Dimension + (absy);
       MPI_Isend(PartCoordsBuff,NPartPerCell*2+1,MPI_DOUBLE,CellMap[absx+1][absy+1][0],
		tag,MPI_COMM_WORLD,&reqs[iscount]);
      iscount++;
    }
    //i+1,j-1
    if(absx!=Dimension-1 && absy!=0){
      int tag = (absx)*Dimension + (absy);
      MPI_Isend(PartCoordsBuff,NPartPerCell*2+1,MPI_DOUBLE,CellMap[absx+1][absy-1][0],
		tag,MPI_COMM_WORLD,&reqs[iscount]);
      iscount++;
    }
    //i-1,j+1
    if(absx!=0 && absy!= Dimension-1){
      int tag = (absx)*Dimension + (absy);
      MPI_Isend(PartCoordsBuff,NPartPerCell*2+1,MPI_DOUBLE,CellMap[absx-1][absy+1][0],
		tag,MPI_COMM_WORLD,&reqs[iscount]);
      iscount++;
    }
    //i-1,j-1
    if(absx!=0 && absy!=0){
      int tag = (absx)*Dimension + (absy);
      MPI_Isend(PartCoordsBuff,NPartPerCell*2+1,MPI_DOUBLE,CellMap[absx-1][absy-1][0],
		tag,MPI_COMM_WORLD,&reqs[iscount]);
      iscount++;
    }
    nreqr = ircount;
    nreqs = iscount;
  }
  void PostWaits(void){
    MPI_Status statr[8];
    MPI_Status stats[8];
    MPI_Waitall(nreqs,reqs,stats);
    MPI_Waitall(nreqr,reqr,statr);
  }
};
    
//Force calculation
//------------------------------------
// Overloaded to handle new case
double interact(Particle &atom1, double atom2x, double atom2y. double atom2z){
  double rx,ry,rz,r,fx,fy,fz,f;
  double sigma6,sigma12;
  
  // computing base values
  rx = atom1.x - atom2x;
  ry = atom1.y - atom2y;
  rz = atom1.z - atom2z;
  r = rx*rx + ry*ry +  rz*rz;

  if(r < 0.000001)
    return 0.0;
  
  r=sqrt(r);
  double sigma2 = (ARsigma/r)*(ARsigma/r);
  sigma6 = sigma2*sigma2*sigma2;
  sigma12 = sigma6*sigma6;
  f = ((sigma12-0.5*sigma6)*48.0*AReps)/r;
  fx = f * rx;
  fy = f * ry;
  fz = f * rz;
  // updating particle properties
  atom1.fx += fx;
  atom1.fy += fy;
  atom1.fz += fz;
  return 4.0*AReps*(sigma12-sigma6);
}

double interact(Particle &atom1, Particle &atom2){
  return interact(atom1,atom2.x,atom2.y,atom2.z);
}

int ComputeAtomsPerCell(int ***CellMap, 
			int NRows,int NCols, 
			int NParts){

  
  int max = NPartPerCell;
  for(int i=0;i<NRows;i++)
    for(int j=0;j<NCols;j++)
      CellMap[i][j][3] = NPartPerCell;
  
  int molsum = NRows*NCols*NPartPerCell;
  while(molsum < NParts){
    max++;
    for(int i=0;i<NRows;i++){
      for(int j=0;j<NCols;j++){
	CellMap[i][j][3]++;
	molsum++;
	if(molsum >= NParts) {
	  return max;
	}
      }
    }
  }
  return max;
}
			 

int main(int argc,char* argv[]){
  
  MPI_Init(&argc,&argv);
  int MY_PE,NCPUS;
  MPI_Comm_rank(MPI_COMM_WORLD,&MY_PE);
  MPI_Comm_size(MPI_COMM_WORLD,&NCPUS);

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
  
  // Integer vectors to specify the six neighbor nodes
  vector<<vector<int> > iv = {
			      {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}
  };
  
  int NumParticles = atoi(argv[1]);
  int NumIterations= atoi(argv[2]);
  
  int Dimension    = int(sqrt(NumParticles/double(NPartPerCell)));
  int TotalCells   = Dimension*Dimension;
  int NCellsPerCPU = 0;
  int MaxCellSize;
  // Variables for the local size of data
  int NLocRows,NLocRowsShort,NLocRowsPE;
  int NLocCols,NLocColsShort,NLocColsPE;
  
  //Figuring out the dimensions of the local data is not trivial
  
  
  // Create a map for the which processor each cell is assigned
  // This map is replicated, but small

  // First Assign the processors
  
  int ***CellCpus = (int***)malloc(sizeof(int**)*Dimension);
  for(int i=0;i<Dimension;i++){
    CellCpus[i] = (int**)malloc(sizeof(int*)*Dimension);
    for(int j=0;j<Dimension;j++)
      CellCpus[i][j] = (int*)calloc(sizeof(int),4); // 0 MPI_Rank the Cell Belongs
                                                    // 1 2 are the local x,y
                                                    // 3 is the number of particles in the cell
  }	
  int icpu = 0;
  int ixend,iyend;
  for(int i=0;i<SQncpus;i++){
    ixend = NLocRows;
    if((((i+1)*NLocRows) > Dimension) || (i==(SQncpus-1))) ixend = NLocRowsShort;
    for(int j=0;j<SQncpus; j++){
      iyend = NLocCols;
      if(((j+1)*NLocCols > Dimension) || (j==(SQncpus-1))) iyend = NLocColsShort;      
      int ixcount =0;
      int iycount =0;
      for(int ix=0;ix<ixend;ix++){
	iycount = 0;
	for(int iy=0;iy<iyend;iy++){
	  CellCpus[i*NLocRows+ix][j*NLocCols+iy][0] = icpu;
	  CellCpus[i*NLocRows+ix][j*NLocCols+iy][1] = ixcount;
	  CellCpus[i*NLocRows+ix][j*NLocCols+iy][2] = iycount;
	  CellCpus[i*NLocRows+ix][j*NLocCols+iy][3] = 0;
	  if(MY_PE == icpu) NCellsPerCPU++;
	  iycount++;
	}
	ixcount++;
      }
      icpu++;
    }
  }
  
  // OK now that we know which cell is on which cpu, we need to allocate the main space
 
  //This is replicated, so that it gives exactly the same number of particles as the
  //serial version
  MaxCellSize = ComputeAtomsPerCell(CellCpus,Dimension,Dimension,NumParticles);
  
  if(MY_PE == 0){
    cout << "\nThe Total Number of Cells is " << TotalCells
	 << " With a maximum of " << MaxCellSize << " particles per cell,"
	 << "\nand "  << NumParticles << " particles total in system\n";
    fflush(stdout);
  }

  NPartPerCell = MaxCellSize;

  // Allocate Cell, space for Particles couched in the Cell creator
  // Local space, only enough for what is on the CPU
  
  vector< vector< Cell > > cells(NLocRowsPE);
  for (int i = 0; i < NLocRowsPE; i++) {
    for (int j = 0; j < NLocColsPE; j++) {
      int ax = ((MY_PE/SQncpus)*NLocRows)+i;
      int ay = ((MY_PE%SQncpus)*NLocCols)+j;
      cells[i].push_back(Cell(i,j,ax,ay,CellCpus[ax][ay][3]));
    }
  }
  
  double TimeStart = MPI_Wtime();
  for (int t = 0; t < NumIterations; t++) { // For each timestep
    double TotPot1=0.0;
    double TotPot2=0.0;
    double TotKin=0.0;
      
    // Start the communications going
    // We can communicate whilst all of the cells are computing the 
    // interactions with themselves.

    for (int cx1 = 0; cx1 < NLocRowsPE; cx1++) {
      for (int cy1 = 0; cy1 < NLocColsPE; cy1++) {
	cells[cx1][cy1].Communicate(CellCpus,Dimension);
      }
    }
 
    // Computing the interactions of the cells with the particles inside.
    for (int cx1 = 0; cx1 < NLocRowsPE; cx1++) {
      for (int cy1 = 0; cy1 < NLocColsPE; cy1++) {
	// Consider interactions between particles within the cell
	for (int i = 0; i < cells[cx1][cy1].particles.size(); i++) {
	  for (int j = 0; j < cells[cx1][cy1].particles.size(); j++) {
	    if(i!=j)
	      TotPot1 += interact(cells[cx1][cy1].particles[i],
				 cells[cx1][cy1].particles[j]);
	  }
	}
      }
    }
    
    // We have to wait here for the comm to finish 
    for(int cx1=0;cx1 < NLocRowsPE;cx1++)
      for (int cy1 = 0; cy1 < NLocColsPE; cy1++) 
    	cells[cx1][cy1].PostWaits();

    // Consider each other cell...
    for(int cx1=0;cx1 < NLocRowsPE;cx1++){
      for(int cy1=0;cy1 < NLocColsPE;cy1++){
	for (int cx2 = 0; cx2 < cells[cx1][cy1].nreqr; cx2++) {
	  // Consider all interactions between particles
	  int nparts=cells[cx1][cy1].remote_particles[cx2][0];
	  for (int i = 0; i < cells[cx1][cy1].particles.size(); i++) {
	    for (int j = 0; j < nparts; j++) {
	      TotPot2 += interact(cells[cx1][cy1].particles[i],
				  cells[cx1][cy1].remote_particles[cx2][j*2+1],
				  cells[cx1][cy1].remote_particles[cx2][j*2+2]);
	    }
	  }
	}
      }
    }
    
    // End iteration over cells    
    // Apply the accumulated forces; update accelerations, velocities, positions
    for (int cx1 = 0; cx1 < NLocRowsPE; cx1++) {
      for (int cy1 = 0; cy1 < NLocColsPE; cy1++) {
	for(int i=0; i < cells[cx1][cy1].particles.size(); i++) {
	  TotKin += cells[cx1][cy1].particles[i].update();
	}
      }
    }
    double TotPotSum = 0.0;
    double TotKinSum = 0.0;
    double LSum = TotPot1+TotPot2;
    
    MPI_Reduce(&LSum,&TotPotSum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&TotKin,&TotKinSum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if(MY_PE == 0)
      printf("\nIteration#%d with Total Energy %e per Atom",
	     t,(TotKinSum+TotPotSum)/NumParticles);
  } // iterate time steps
  double TimeEnd = MPI_Wtime();
  if(MY_PE == 0)
    cout << "\nTime for " << NumIterations << " is "<< TimeEnd-TimeStart;
  
  for(int i=0;i<Dimension;i++){
    for(int j=0;j<Dimension;j++){
      free(CellCpus[i][j]);
    }
    free(CellCpus[i]);
  }
  free(CellCpus);

  MPI_Finalize();
  return 0;
}
