/*----------------------------------------------------------------------
Program pmd.cpp performs parallel molecular-dynamics for Lennard-Jones
systems using the Message Passing Interface (MPI) standard.
----------------------------------------------------------------------*/
#include <iostream>
#include <cmath>
#include <fstream>      // std::fstream
#include <vector>
#include <array>
#include <algorithm>
#include <map>
#include "mpi.h"
#include "pmd.hpp"

/* Constants for the random number generator */
#define D2P31M 2147483647.0
#define DMUL 16807.0

const double RCUT = 2.5; // Potential cut-off length
const double MOVED_OUT = -1.0e10;

using namespace std;

// Class for keeping track of the properties for an atom
class Atom {
public:
  double type;             // identifier for atom type
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
  Atom() 
    : type(0),isResident(true),x(0.0),y(0.0),z(0.0),
      ax(0.0),ay(0.0),az(0.0),vx(0.0),vy(0.0),vz(0.0){
  }
};
 

class SubSystem {
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
  SubSystem(int sid, array<int, 3> vproc, array<int, 3> InitUcell, double InitTemp, double Density): pid(0), n(0), nglob(0), comt(0.0), al{}, vid{}, myparity{}, nn{}, sv{}, vSum{}, gvSum{}, atoms{}, kinEnergy(0.0), potEnergy(0.0), totEnergy(0.0), temperature(0) {

    pid = sid;

    /* Compute basic parameters */
    for (int i=0; i<3; i++) al[i] = InitUcell[i]/cbrt(Density/4.0);
    // if (pid == 0) cout << "al = " << al[0] << " " << al[1] <<  " " << al[2] << endl;

    // Prepare the Neighbot-node table
    InitNeighborNode(vproc);

    // Initialize lattice positions and assign random velocities
    array<double, 3> c{},gap{};
    int j,a,nX,nY,nZ;

    /* FCC atoms in the original unit cell */
    vector<vector<double> > origAtom = {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.5},
					{0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};

    /* Set up a face-centered cubic (fcc) lattice */
    for (a=0; a<3; a++) gap[a] = al[a]/InitUcell[a];

    for (nZ=0; nZ<InitUcell[2]; nZ++) {
      c[2] = nZ*gap[2];
      for (nY=0; nY<InitUcell[1]; nY++) {
	c[1] = nY*gap[1];
	for (nX=0; nX<InitUcell[0]; nX++) {
	  c[0] = nX*gap[0];
	  for (j=0; j<4; j++) {
	    Atom atom;

	    atom.x = c[0] + gap[0]*origAtom[j][0];

	    atom.y = c[1] + gap[1]*origAtom[j][1];

	    atom.z = c[2] + gap[2]*origAtom[j][2];
	    // if(pid == 0) cout << "atom coordinates - " << atom.x << " " << atom.y << " " << atom.z << endl;

	    atoms.push_back(atom);	    
	  }
	}
      }
    }
    /* Total # of atoms summed over processors */
    n = atoms.size();
    MPI_Allreduce(&n, &nglob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    /* Generate random velocities */
    double  seed = 13597.0+sid;
    double vMag = sqrt(3*InitTemp);
    double e[3];
    for(a=0; a<3; a++) vSum[a] = 0.0;
    for(auto & atom : atoms) {
      RandVec3(e,&seed);
      atom.vx = vMag*e[0];
      vSum[0] += atom.vx;
      atom.vy = vMag*e[1];
      vSum[1] += atom.vy;
      atom.vz = vMag*e[2];
      vSum[2] += atom.vz;
    }
    MPI_Allreduce(&vSum[0],&gvSum[0],3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    // Make the total momentum zero
    for (a=0; a<3; a++) gvSum[a] /= nglob;
    for(auto & atom : atoms) {
      atom.vx -= gvSum[0];
      atom.vy -= gvSum[1];
      atom.vz -= gvSum[2];
    }
  }

  void InitNeighborNode(array<int, 3> vproc) {
    // Prepare neighbor-node ID table for a subsystem
    vid[0] = pid/(vproc[1]*vproc[2]);
    vid[1] = (pid/vproc[2])%vproc[1];
    vid[2] = pid%vproc[2];
    
    vector<vector<int > > iv = {
				{-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}
    };
    
    int ku, a, k1[3];
    
    /* Set up neighbor tables, nn & sv */
    for (ku=0; ku<6; ku++) {
      /* Vector index of neighbor ku */
      for (a=0; a<3; a++)
	k1[a] = (vid[a]+iv[ku][a]+vproc[a])%vproc[a];
      /* Scalar neighbor ID, nn */
      nn[ku] = k1[0]*vproc[1]*vproc[2]+k1[1]*vproc[2]+k1[2];
      /* Shift vector, sv */
      vector<double> svEntry;
      for (a=0; a<3; a++) svEntry.push_back(al[a]*iv[ku][a]);
      sv.push_back(svEntry);
    }
    
    // Set up node parity table
    for (a=0; a<3; a++) {
      if (vproc[a] == 1) 
	myparity[a] = 2;
      else if (vid[a]%2 == 0)
	myparity[a] = 0;
      else
	myparity[a] = 1;
    }
  }

  // Update Atomic co-ordinates to r(t+Dt)
  void Update(double DeltaT) {
    for(auto & atom : atoms) {
      atom.x = atom.x + DeltaT*atom.vx;
      atom.y = atom.y + DeltaT*atom.vz;
      atom.z = atom.z +DeltaT*atom.vy;
    }
  }

  // Update the velocities after a time-step DeltaT
  void Kick(double DeltaT) {
    for (auto & atom : atoms) {
      atom.vx = atom.vx + DeltaT*atom.ax;
      atom.vy = atom.vy + DeltaT*atom.ay;
      atom.vz = atom.vz +DeltaT*atom.az;
    }
  }

  // Exchange boundaty-atom co-ordinates among neighbor nodes
  void AtomCopy() {
    int kd,kdd,i,ku,inode,nsd,nrc,a;
    int nbnew = 0; /* # of "received" boundary atoms */
    double com1 = 0;
    vector<vector<int> > lsb (6);

    remove_if(atoms.begin(), atoms.end(), [](Atom atom) {return !atom.isResident;});
    /* Main loop over x, y & z directions starts--------------------------*/
    for (kd=0; kd<3; kd++) {

      // Iterate through all atoms in this cell
      for(auto it_atom = atoms.begin(); it_atom != atoms.end(); ++it_atom ) {
	for (kdd=0; kdd<2; kdd++) {
	  ku = 2*kd+kdd; /* Neighbor ID */
	  /* Add an atom to the boundary-atom list, LSB, for neighbor ku 
	     according to bit-condition function, bbd */
	  i = distance(atoms.begin(), it_atom);
	  if (bbd(*it_atom,ku)) lsb[ku].push_back(i);	 
	}       
      }
      if(pid ==0) cout << "atoms identified for copy to " << (ku -1)<< " & " << ku << " : " << lsb[ku-1].size() << " " << lsb[ku].size() << endl;
      // if(pid == 0)cout << "atoms searched as far as " << i << endl;
      /* Message passing------------------------------------------------*/   
      
      com1=MPI_Wtime(); /* To calculate the communication time */
      
      /* Loop over the lower & higher directions */
      for (kdd=0; kdd<2; kdd++) {
	
	vector<double> sendBuf;
	vector<double> recvBuf;
	
	ku=2*kd+kdd;
	// if(pid ==0) cout << "ku = " << ku << endl;
	inode = nn[ku]; /* Neighbor node ID */
	// if(pid == 0) cout << "inode = " << inode << endl;
	
	nsd = lsb[ku].size(); 
	// cout << "copy number " << nsd << endl;
	
	/* Even node: send & recv */
	if (myparity[kd] == 0) {
	  MPI_Send(&nsd,1,MPI_INT,inode,10,MPI_COMM_WORLD);
	  MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,10,
		   MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	}
	/* Odd node: recv & send */
	else if (myparity[kd] == 1) {
	  MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,10,
		   MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	  MPI_Send(&nsd,1,MPI_INT,inode,10,MPI_COMM_WORLD);
	}
	/* Single layer: Exchange information with myself */
	else
	  nrc = nsd;

	// if(pid == 0) cout << "nrc = " << nrc /4<< endl;
	/* Now nrc is the # of atoms to be received */
	
	/* Send & receive information on boundary atoms-----------------*/
	
	/* Message buffering */
	for (auto it_index = lsb[ku].begin(); it_index != lsb[ku].end(); ++it_index) {
	  sendBuf.push_back(atoms[*it_index].type);
	  sendBuf.push_back(atoms[*it_index].x - sv[ku][0]);
	  // if(pid == 0) cout << "copy - " << atoms[*it_index].x - sv[ku][0] << " ";
	  sendBuf.push_back(atoms[*it_index].y - sv[ku][1]);
	  // if(pid ==0) cout << atoms[*it_index].y - sv[ku][1] << " ";
	  sendBuf.push_back(atoms[*it_index].z - sv[ku][2]);
	  // if(pid ==0) cout << atoms[*it_index].z - sv[ku][2] << " " << endl;
	}		
	// resize the receive buffer for nrc
	recvBuf.resize(4*nrc);
      
	/* Even node: send & recv */
	if (myparity[kd] == 0) {
	  MPI_Send(&sendBuf[0],4*nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
	  MPI_Recv(&recvBuf[0],4*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
		   MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	}
	/* Odd node: recv & send */
	else if (myparity[kd] == 1) {
	  MPI_Recv(&recvBuf[0],4*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
		   MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	  MPI_Send(&sendBuf[0],4*nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
	}
	/* Single layer: Exchange information with myself */
	else
	  sendBuf.swap(recvBuf);

	// Message storing
	for(i = 0; i < 4*nrc ; i++) {
	  Atom rAtom;

	  rAtom.type = recvBuf[i];
	  ++i;
	  rAtom.isResident = false;
	  rAtom.x =  recvBuf[i];
	  ++i;
	  rAtom.y =  recvBuf[i];
	  ++i;
	  rAtom.z =  recvBuf[i];
	  // if(pid == 0) cout << " arriving in atom copy - x: " << rAtom.x;
	//   rAtom.x = *it_recv;
	//   ++it_recv;
	  // if(pid == 0) cout << " y: " << rAtom.y;
	//   rAtom.y = *it_recv;
	//   ++it_recv;
	  // if(pid == 0) cout << " z: " << rAtom.z << endl;
	//   rAtom.z = *it_recv;
	
	  atoms.push_back(rAtom);	  
	}
	/* Internode synchronization */
	MPI_Barrier(MPI_COMM_WORLD);
      
      } /* Endfor lower & higher directions, kdd */
      
      comt += MPI_Wtime()-com1; /* Update communication time, COMT */
    }
  }

  // Send moved-out atoms to neighbor nodes and receive moved-in atoms
  // from neighbor nodes
  void AtomMove() {
    vector<vector<int> > mvque (6); 
    int ku,kd,i,kdd,kul,kuh,inode,ipt,a,nsd,nrc;
    int newim = 0; /* # of new immigrants */
    double com1 = 0;  

    /* Main loop over x, y & z directions starts------------------------*/

  for (kd=0; kd<3; kd++) {

        /* Make a moved-atom list, mvque----------------------------------*/

    /* Scan all the residents & immigrants to list moved-out atoms */
    for(auto it_atom = atoms.begin(); it_atom != atoms.end(); ++it_atom ) {
      kul = 2*kd  ; /* Neighbor ID */
      kuh = 2*kd+1; 
      /* Register a to-be-copied atom in mvque[kul|kuh][] */      
      if (it_atom->x > MOVED_OUT) { /* Don't scan moved-out atoms */
        /* Move to the lower direction */
	i = distance(atoms.begin(), it_atom);
	if(it_atom->isResident) {
	  if (bmv(*it_atom,kul)) mvque[kul].push_back(i);
        /* Move to the higher direction */
	  else if (bmv(*it_atom,kuh)) mvque[kuh].push_back(i);
	}
      }
    }
    if(pid ==0) cout << "atoms identified for move to : " << kul << " & " << kuh << " : " << mvque[kul].size() << " " << mvque[kuh].size() << endl;
    /* Message passing------------------------------------------------*/   
      
    com1=MPI_Wtime(); /* To calculate the communication time */
      
    /* Loop over the lower & higher directions */
    for (kdd=0; kdd<2; kdd++) {
      // if(pid ==0) cout << "kdd " << kdd << endl;
      vector<double> sendBuf;
      vector<double> recvBuf;
	
      ku=2*kd+kdd;
      // if(pid ==0) cout << "ku = " << ku << endl;
      inode = nn[ku]; /* Neighbor node ID */
      // if(pid == 0) cout << "inode = " << inode << endl;
	
      nsd = mvque[ku].size(); 
      // cout << "copy number " << nsd << endl;
	
      /* Even node: send & recv */
      if (myparity[kd] == 0) {
	MPI_Send(&nsd,1,MPI_INT,inode,10,MPI_COMM_WORLD);
	MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,10,
		 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
      /* Odd node: recv & send */
      else if (myparity[kd] == 1) {
	MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,10,
		 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	MPI_Send(&nsd,1,MPI_INT,inode,10,MPI_COMM_WORLD);
      }
      /* Single layer: Exchange information with myself */
      else
	nrc = nsd;

      /* Now nrc is the # of atoms to be received */
      
      /* Send & receive information on boundary atoms-----------------*/
      
      /* Message buffering */
      for (auto it_index = mvque[ku].begin(); it_index != mvque[ku].end(); ++it_index) {
	sendBuf.push_back(atoms[*it_index].type);
	
	sendBuf.push_back(atoms[*it_index].x - sv[ku][0]);
	sendBuf.push_back(atoms[*it_index].y - sv[ku][1]);
	sendBuf.push_back(atoms[*it_index].z - sv[ku][2]);
	// In AtomMove we will also be considering the velocities
	sendBuf.push_back(atoms[*it_index].vx);
	sendBuf.push_back(atoms[*it_index].vy);
	sendBuf.push_back(atoms[*it_index].vz);

	
	//atoms[*it_index].isResident = false;
	// Mark the atom as moved out
	atoms[*it_index].x = MOVED_OUT;
      }
               
      // resize the receive buffer for nrc
      recvBuf.resize(7*nrc);
      
      /* Even node: send & recv */
      if (myparity[kd] == 0) {
	MPI_Send(&sendBuf[0],7*nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
	MPI_Recv(&recvBuf[0],7*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
		 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
      /* Odd node: recv & send */
      else if (myparity[kd] == 1) {
	MPI_Recv(&recvBuf[0],7*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
		 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	MPI_Send(&sendBuf[0],7*nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
      }
	/* Single layer: Exchange information with myself */
      else
	sendBuf.swap(recvBuf);

      // Message storing
      for(i = 0; i < 7*nrc ;i++) {
	Atom rAtom;
	
	rAtom.type = recvBuf[i];
	++i;
	rAtom.isResident = true;
	rAtom.x = recvBuf[i];
	++i;
	rAtom.y = recvBuf[i];
	++i;
	rAtom.z = recvBuf[i];
	++i;
	rAtom.vx = recvBuf[i];
	++i;
	rAtom.vy = recvBuf[i];
	++i;
	rAtom.vz = recvBuf[i];	

	atoms.push_back(rAtom);
      }
      /* Internode synchronization */
      MPI_Barrier(MPI_COMM_WORLD);
    } /* Endfor lower & higher directions, kdd */

    comt += MPI_Wtime()-com1; /* Update communication time, COMT */
  } /* Endfor x, y & z directions, kd */

  /* Main loop over x, y & z directions ends--------------------------*/

  /* Compress resident arrays including new immigrants */

  
  remove_if(atoms.begin(), atoms.end(), 
  	    [](Atom atom) { return atom.x >= MOVED_OUT; });
  
  }
  
  // Return true if an Atom lies in them boundary to a neighbor ID
  int bbd(Atom atom, int ku) {
    // if (atom.isResident == false) return false; // Do not consider atoms that have moved already 
    int kd,kdd;
    kd = ku/2; /* x(0)|y(1)|z(2) direction */
    kdd = ku%2; /* Lower(0)|higher(1) direction */
    if (kdd == 0){
      if (kd == 0)
	return atom.x < RCUT;
      if (kd == 1)
	return atom.y < RCUT;
      if (kd == 2)
	return atom.z < RCUT;
    }
    else {
      if (kd == 0)
	return al[0]-RCUT < atom.x;
      if (kd == 1)
	return al[1]-RCUT < atom.y;
      if (kd == 2)
	return al[2]-RCUT < atom.z;
    }
    return 0; // default return
  }

  // Return true if an Atom lies in them boundary to a neighbor ID
  int bmv(Atom atom, int ku) {    
    int kd,kdd;
    kd = ku/2; /* x(0)|y(1)|z(2) direction */
    kdd = ku%2; /* Lower(0)|higher(1) direction */
    if (kdd == 0){
      if (kd == 0)
	return atom.x < 0.0;
      if (kd == 1)
	return atom.y < 0.0;
      if (kd == 2)
	return atom.z < 0.0;
    }
    else {
      if (kd == 0)
	return al[0] < atom.x;
      if (kd == 1)
	return al[1] < atom.y;
      if (kd == 2)
	return al[2] < atom.z;
    }
    return 0;
  }

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

  /*--------------------------------------------------------------------*/
  void EvalProps(int stepCount, double DeltaT) {
    /*----------------------------------------------------------------------
      Evaluates physical properties: kinetic, potential & total energies.
      ----------------------------------------------------------------------*/
    double vv=0,lke=0;
    int i,a;

    /* Total kinetic energy */
    for(auto & atom : atoms) {
        vv = (atom.vx*atom.vx + atom.vy*atom.vy + atom.vz*atom.vz);
        lke += vv;
    }
    lke *= 0.5;
    MPI_Allreduce(&lke,&kinEnergy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    /* Energy per atom */
    kinEnergy /= nglob;
    potEnergy /= nglob;
    totEnergy = kinEnergy + potEnergy;
    temperature = kinEnergy*2.0/3.0;

    /* Print the computed properties */
    if (pid == 0) cout << stepCount*DeltaT << " " << lke << " " << temperature << " " << potEnergy << " " << totEnergy <<endl;
  }
};

/*--------------------------------------------------------------------*/
int main(int argc, char **argv) {
  /*--------------------------------------------------------------------*/
  // cpu - Elapsed wall clock time in seconds
  double cpu,cpu1;

  int sid; // Sequential processor ID
  MPI_Init(&argc,&argv); /* Initialize the MPI environment */
  MPI_Comm_rank(MPI_COMM_WORLD, &sid);  /* My processor ID */

  // vproc - number of processrs in x|y|z directions
  // Initucell - Number of unit cells per processor
  // Density - Density of atoms
  // InitTemp - Starting temperature
  // DeltaT - Size of time step
  // StepLimit - Number of time steps to be simulated
  // StepAvg - Reporting interval forces statistical data
  array<int, 3> vproc{}, InitUcell{};
  double Density, InitTemp, DeltaT;
  int StepLimit, StepAvg;

  /* Read control parameters */
  ifstream ifs("pmd.in", ifstream::in);
  if(!ifs.is_open()) {
    cerr << "failed to open input file" << endl;
    terminate();
  }

  ifs >> vproc[0] >> vproc[1] >> vproc[2];
  ifs >> InitUcell[0] >> InitUcell[1] >> InitUcell[2];
  ifs >> Density >> InitTemp >> DeltaT;
  ifs >> StepLimit >> StepAvg;

  ifs.close();

  SubSystem subsystem(sid, vproc, InitUcell, InitTemp, Density);
  if(sid == 0) cout << "nglob = " << subsystem.nglob << endl;
  subsystem.AtomCopy();
  ComputeAccel(subsystem);

  cpu1 = MPI_Wtime();
  for (int stepCount=1; stepCount<=StepLimit; stepCount++) {
    SingleStep(subsystem, DeltaT);    
    if (stepCount%StepAvg == 0) subsystem.EvalProps(stepCount, DeltaT);
  }
  cpu = MPI_Wtime() - cpu1;
  if (sid == 0) cout << "CPU & COMT = " << cpu << " " << subsystem.comt << endl;

  MPI_Finalize(); /* Clean up the MPI environment */
  return 0;
}

/*--------------------------------------------------------------------*/
void SingleStep(SubSystem &subsystem, double DeltaT) {
/*----------------------------------------------------------------------
r & rv are propagated by DeltaT using the velocity-Verlet scheme.
----------------------------------------------------------------------*/
  double DeltaTH = DeltaT / 2.0;
  subsystem.Kick(DeltaTH); /* First half kick to obtain v(t+Dt/2) */
  subsystem.Update(DeltaT);
  subsystem.AtomMove();
  subsystem.AtomCopy();
  ComputeAccel(subsystem); /* Computes new accelerations, a(t+Dt) */
  subsystem.Kick(DeltaTH); /* Second half kick to obtain v(t+Dt) */
}

/*--------------------------------------------------------------------*/
void ComputeAccel(SubSystem &subsystem) {
  /*----------------------------------------------------------------------
    Given atomic coordinates, r[0:n+nb-1][], for the extended (i.e., 
    resident & copied) system, computes the acceleration, ra[0:n-1][], for 
    the residents.
    ----------------------------------------------------------------------*/
  int i,j,a,lc2[3],lcyz2,lcxyz2,mc[3],c,mc1[3],c1;
  int bintra;
  double dr[3],rr,ri2,ri6,r1,rrCut,fcVal,f,vVal,lpe;

  double Uc, Duc;

  array<int, 3> lc{};
  array<double, 3> rc{};

  //vector<int> head;
  map<int, int> head;
  vector<int> lscl (subsystem.atoms.size());
  int EMPTY = -1;
  
  /* Compute the # of cells for linked cell lists */
  for (a=0; a<3; a++) {
    lc[a] = subsystem.al[a]/RCUT; 
    rc[a] = subsystem.al[a]/lc[a];
  }
  // if (subsystem.pid == 0) {
  //   cout << "lc = " << lc[0] << " " << lc[1] << " " << lc[2] << endl;
  //   cout << "rc = " << rc[0] << " " << rc[1] << " " << rc[2] << endl;
  // }
  
  /* Constants for potential truncation */
  rr = RCUT*RCUT; ri2 = 1.0/rr; ri6 = ri2*ri2*ri2; r1=sqrt(rr);
  Uc = 4.0*ri6*(ri6 - 1.0);
  Duc = -48.0*ri6*(ri6 - 0.5)/r1;

  /* Reset the potential & forces */
  lpe = 0.0;
  for(auto & atom : subsystem.atoms) {
    atom.ax = 0.0;
    atom.ay = 0.0;
    atom.az = 0.0;
  }

  /* Make a linked-cell list, lscl------------------------------------*/

  for (a=0; a<3; a++) lc2[a] = lc[a]+2;
  lcyz2 = lc2[1]*lc2[2];
  lcxyz2 = lc2[0]*lcyz2;

  /* Reset the headers, head */
  //for (c=0; c<lcxyz2; c++) head.push_back(EMPTY);

  /* Scan atoms to construct headers, head, & linked lists, lscl */
  if(subsystem.pid == 0) cout << "atoms in subsystem  = "  << subsystem.atoms.size() << endl;
  for (auto it_atom = subsystem.atoms.begin(); it_atom != subsystem.atoms.end(); ++it_atom) {
    mc[0] = (int)(it_atom->x + rc[0]) / rc[0];
    mc[1] = (int)(it_atom->y + rc[1]) / rc[1];
    mc[2] = (int)(it_atom->z + rc[2]) / rc[2];
    /* Translate the vector cell index, mc, to a scalar cell index */
    c = mc[0]*lcyz2+mc[1]*lc2[2]+mc[2];

    /* Link to the previous occupant (or EMPTY if you're the 1st) */
    //if(subsystem.pid == 0) cout << "before head size: " << head.size() << " c: "<< c << " i: " << i <<  " head[c]: " << head[c] << endl;
    auto search = head.find(c);
    if(search != head.end())
      lscl[distance(subsystem.atoms.begin(), it_atom)] = search->second;
    else
      lscl[distance(subsystem.atoms.begin(), it_atom)] = EMPTY;
    //if(subsystem.pid == 0) cout <<"after head[c]: " << head[c] << endl;
    //continue;
    /* The last one goes to the header */
    if(search != head.end())
      head.erase(search);
    head.insert(head.begin(), pair<int, int>(c,distance(subsystem.atoms.begin(), it_atom)));
    //if(subsystem.pid == 0) cout << "You have reached the loop end" << endl;
  } /* Endfor atom i */

  /* Calculate pair interaction---------------------------------------*/
  rrCut = RCUT*RCUT;

  /* Scan inner cells */
  for (mc[0]=1; mc[0]<=lc[0]; (mc[0])++)
    for (mc[1]=1; mc[1]<=lc[1]; (mc[1])++)
      for (mc[2]=1; mc[2]<=lc[2]; (mc[2])++) {
	/* Calculate a scalar cell index */
	c = mc[0]*lcyz2+mc[1]*lc2[2]+mc[2];
	/* Skip this cell if empty */
	if (head.find(c) == head.end()) continue;

	/* Scan the neighbor cells (including itself) of cell c */
	for (mc1[0]=mc[0]-1; mc1[0]<=mc[0]+1; (mc1[0])++)
	  for (mc1[1]=mc[1]-1; mc1[1]<=mc[1]+1; (mc1[1])++)
	    for (mc1[2]=mc[2]-1; mc1[2]<=mc[2]+1; (mc1[2])++) {

	      /* Calculate the scalar cell index of the neighbor cell */
	      c1 = mc1[0]*lcyz2+mc1[1]*lc2[2]+mc1[2];
	      /* Skip this neighbor cell if empty */
	      if (head.find(c1) == head.end()) continue;

	      /* Scan atom i in cell c */
	      i = head[c];
	      while (i != EMPTY) {

		/* Scan atom j in cell c1 */
		j = head[c1];
		while (j != EMPTY) {

		  /* No calculation with itself */
		  if (j != i) {
		    /* Logical flag: intra(true)- or inter(false)-pair atom */
		    bintra = (j < subsystem.n);

		    /* Pair vector dr = r[i] - r[j] */
		    dr[0] = subsystem.atoms[i].x - subsystem.atoms[j].x;
		    dr[1] = subsystem.atoms[i].y - subsystem.atoms[j].y;
		    dr[2] = subsystem.atoms[i].z - subsystem.atoms[j].z;
		    for (rr=0.0, a=0; a<3; a++)
		      rr += dr[a]*dr[a];	    

		    /* Calculate potential & forces for intranode pairs (i < j)
		       & all the internode pairs if rij < RCUT; note that for
		       any copied atom, i < j */
		    if (i<j && rr<rrCut) {
		      ri2 = 1.0/rr; ri6 = ri2*ri2*ri2; r1 = sqrt(rr);
		      fcVal = 48.0*ri2*ri6*(ri6-0.5) + Duc/r1;
		      vVal = 4.0*ri6*(ri6-1.0) - Uc - Duc*(r1-RCUT);
		      if (bintra) lpe += vVal; else lpe += 0.5*vVal;

		      f = fcVal*dr[0];
		      subsystem.atoms[i].ax += f;
		      if(bintra) subsystem.atoms[j].ax -= f;
	      
		      f = fcVal*dr[1];
		      subsystem.atoms[i].ay += f;
		      if(bintra) subsystem.atoms[j].ay -= f;
	      
		      f = fcVal*dr[0];
		      subsystem.atoms[i].az += f;
		      if(bintra) subsystem.atoms[j].az -= f;	                   
		    }
		  } /* Endif not self */
          
		  j = lscl[j];
		} /* Endwhile j not empty */

		i = lscl[i];
	      } /* Endwhile i not empty */

	    } /* Endfor neighbor cells, c1 */

      } /* Endfor central cell, c */

  /* Global potential energy */
  if(subsystem.pid == 0) cout << "local potential energy " << lpe << endl;
  MPI_Allreduce(&lpe,&subsystem.potEnergy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}
