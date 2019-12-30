/*----------------------------------------------------------------------
Program pmd.c performs parallel molecular-dynamics for Lennard-Jones 
systems using the Message Passing Interface (MPI) standard.
----------------------------------------------------------------------*/
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>      // std::fstream
#include <vector>
#include <array>
#include <random>
#include "mpi.h"

const double ARmass = 39.94800000; //A.U.s
const double ARsigma = 3.40500000; // Angstroms
const double AReps   = 119.800000; // Kelvins
const double CellDim = 12.0000000; // Angstroms

const double RCUT = 2.5; // Potential cut-off length

using namespace std;

// Class for keeping track of the properties for an atom
class Atom{
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

  array<int, 3> myParity; // Parity of this processor
  array<int, 6> nn; // Neighbor node list of this processor
  vector<int> cellList; // Scalar cell index list for the subsystem
  vector<vector<double> > sv; // Shift vector to the 6 neighbors
  array<double, 3> vSum, gvSum;
  vector<Atom> systemAtoms; // all atoms within the processor
  int n; // Number of resident atoms in this processor
  double comt; // elapsed wall clock time & Communication time in second
  
  /* Create cell with by taking the parameters for InitUcell and InitTemp 
     we calculate the number of atoms and give them random velocities */
  SubSystem(array<int, 6> neighborNode, array<int, 3> parityTable, sid){
    nn = neighborNode;
    myParity = parityTable;
    pid = sid;

    default_random_engine generator;
    normal_distribution<double> distribution(1,1);
    
    /* Compute basic parameters */
    for (int i=0; i<3; i++) al[i] = InitUcell[i]/cbrt(Density/4.0);
    if (pid == 0) printf("al = %e %e %e\n",al[0],al[1],al[2]);

    // Initialize lattice positions and assign random velocities
    array<double, 3> c,gap;
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
	    atom.vx = sqrt(3*InitTemp)*distribution(generator);
	    atom.y = c[1] + gap[1]*origAtom[j][1];
	    atom.vy = sqrt(3*InitTemp)*distribution(generator);
	    atom.z = c[2] + gap[2]*origAtom[j][2];
	    atom.vz = sqrt(3*InitTemp)*drand48();
	    systemAtoms.push_back(atom);

	    vSum[0] += atom.vx;
	    vSum[1] += atom.vy;
	    vSum[2] += atom.vz;
	  }
	}
      }
    }

    n = atoms.size();
    int nglob; // Total number of atoms summed over processors
    MPI_Allreduce(&n, &nglob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    MPI_Allreduce(&vSum[0],&gvSum[0],3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    
    // Make the total momentum zero
    for (a=0; a<3; a++) gvSum[a] /= nglob;
    for(auto & atom : atoms) {
      atom.vx -= gvSum[0];
      atom.vy -= gvSum[1];
      atom.vz -= gvSum[2];
    }
  }

  // Update the velocities after a time-step DeltaT
  void HalfKick(double DeltaT) {
    for (auto & atom : atoms) {
      atom.vx += DeltaT*atom.ax;
      atom.vy += DeltaT*atom.ay;
      atom.vz += DeltaT*atom.az;
    }
  }

  // Exchange boundaty-atom co-ordinates among neighbor nodes
  void AtomCopy() {
    int kd,kdd,i,ku,inode,nsd,nrc,a;
    int nbnew = 0; /* # of "received" boundary atoms */
    double com1;

    vector<vector<Atom> > lsb; // atom to be send to the

    // Iterate through neighbour nodes
    for( auto it_neighbor = nn.begin(); it_neighbor != nn.end(); ++it_neighbor) {
      // Iterate through all atoms in this cell
      vector<double> sendBuf;
      vector<double> recvBuf;
      for( auto it_atom = atoms.begin(); it_atom != atoms.end(); ++it_atom) {
	if(bbd(*it_atom, *it_neighbor)) {
	  sendBuf.push_back(it_atom->type);
	  sendBuf.push_back(it_atom->x);
	  sendBuf.push_back(it_atom->y);
	  sendBuf.push_back(it_atom->z);
	  it_atom->isResident = false;
	}
      }      
      
      /* Message passing------------------------------------------------*/

      // the first two neighbors need a x - parity check and so on
      if(distance(nn.begin(), it_neighbor) < 2)
	kd = 0;
      else if(distance(nn.begin(), it_neighbor) < 4)
	kd =1;
      else
	kd =2;
      
      com1=MPI_Wtime(); /* To calculate the communication time */

      nsd = sendBuf.size(); /* # of atoms to be sent */
      
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

      // resize the receive buffer for nrc
      recvBuf.resize(nrc);
      
      /* Even node: send & recv */
      if (myparity[kd] == 0) {
	MPI_Send(&sendBuf[0],nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
	MPI_Recv(&recvBuf[0],nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
		 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
      /* Odd node: recv & send */
      else if (myparity[kd] == 1) {
	MPI_Recv(&recvBuf[0],nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
		 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	MPI_Send(&sendBuf[0],nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
      }
	/* Single layer: Exchange information with myself */
      else
	sendBuf.swap(recvBuf);

      // Message storing
      for(auto it_recv = recvBuf.begin(); it_recv != recvBuf.end(); ++it_recv) {
	Atom rAtom;
	
	rAtom.type = *it_recv;
	++it_recv;
	rAtom.isResident = true;
	rAtom.x = *it_recv;
	++it_recv;
	rAtom.y = *it_recv;
	++it_recv;
	rAtom.z = *it_recv;

	atoms.push_back(rAtom);       
      }
      
      // Delete sent message after the step finishes
           
      /* Internode synchronization */
      MPI_Barrier(MPI_COMM_WORLD);
      
    } /* Endfor lower & higher directions, kdd */

    comt += MPI_Wtime()-com1; /* Update communication time, COMT */
  }

  // Send moved-out atoms to neighbor nodes and receive moved-in atoms
  // from neighbor nodes
  void AtomMove() {
    int ku,kd,i,kdd,kul,kuh,inode,ipt,a,nsd,nrc;
    int newim = 0; /* # of new immigrants */
    double com1;

    // Neglect the atoms in them cell that have entered throgh AtomCopy()
    atoms.resize(n);
    
    // Iterate through neighbour nodes
    for(auto it_neighbor = nn.begin(); it_neighbor != nn.end(); ++it_neighbor) {
      // Iterate through all atoms in this cell
      vector<double> sendBuf;
      vector<double> recvBuf;
      for(auto it_atom = atoms.begin(); it_atom != atoms.end(); ++it_atom) {
	if(bmv(*it_atom, *it_neighbor)) {
	  sendBuf.push_back(it_atom->type);
	  sendBuf.push_back(it_atom->x);
	  sendBuf.push_back(it_atom->y);
	  sendBuf.push_back(it_atom->z);
	  // In AtomMove we will also be considering the velocities
	  sendBuf.push_back(it_atom->vx);
	  sendBuf.push_back(it_atom->vy);
	  sendBuf.push_back(it_atom->vz);
	  
	  it_atom->isResident = false;
	}
      }      
      
      /* Message passing------------------------------------------------*/

      // the first two neighbors need a x - parity check and so on
      if(distance(nn.begin(), it_neighbor) < 2)
	kd = 0;
      else if(distance(nn.begin(), it_neighbor) < 4)
	kd =1;
      else
	kd =2;
      
      com1=MPI_Wtime(); /* To calculate the communication time */

      nsd = sendBuf.size(); /* # of atoms to be sent */
      
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

      // resize the receive buffer for nrc
      recvBuf.resize(nrc);
      
      /* Even node: send & recv */
      if (myparity[kd] == 0) {
	MPI_Send(&sendBuf[0],nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
	MPI_Recv(&recvBuf[0],nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
		 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
      /* Odd node: recv & send */
      else if (myparity[kd] == 1) {
	MPI_Recv(&recvBuf[0],nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
		 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	MPI_Send(&sendBuf[0],nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
      }
	/* Single layer: Exchange information with myself */
      else
	sendBuf.swap(recvBuf);

      // Message storing
      for(auto it_recv = recvBuf.begin(); it_recv != recvBuf.end(); ++it_recv) {
	Atom rAtom;
	
	rAtom.type = *it_recv;
	++it_recv;
	rAtom.isResident = true;
	rAtom.x = *it_recv;
	++it_recv;
	rAtom.y = *it_recv;
	++it_recv;
	rAtom.z = *it_recv;
	++it_recv;
	rAtom.vx = *it_recv;
	++it_recv;
	rAtom.vy = *it_recv;
	++it_recv;
	rAtom.vz = *it_recv;	

	atoms.push_back(rAtom);       
      }
      
      // Delete sent message after the step finishes
           
      /* Internode synchronization */
      MPI_Barrier(MPI_COMM_WORLD);
      
    } /* Endfor lower & higher directions, kdd */

    atoms.erase(remove_if(atoms.begin(), atoms.end(), 
                       [](Atom atom) { return !atom.isResident; }), atoms.end());
    n = atoms.size();

    comt += MPI_Wtime()-com1; /* Update communication time, COMT */
  }
  
  // Return true if an Atom lies in them boundary to a neighbor ID
  int bbd(Atom atom, int ku) {
    if (atom.isResident == 0) return 0; // Do not consider atoms that have moved already 
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
  }

  // Return true if an Atom lies in them boundary to a neighbor ID
  int bmv(Atom atom, int ku) {
    if (atom.isResident == 0) return 0; // Do not consider atoms that have moved already 
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
  }
};

/*--------------------------------------------------------------------*/
int main(int argc, char **argv) {
/*--------------------------------------------------------------------*/
  double cpu1;

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
  array<int, 3> vproc, InitUcell; 
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

  array<int, 3> vid{}; /* Vector index of this processor */
  vid[0] = pid/(vproc[1]*vproc[2]);
  vid[1] = (pid/vproc[2])%vproc[1];
  vid[2] = pid%vproc[2];
  
  /* Compute basic parameters */
  Double DeltaTH = 0.5*DeltaT; // Half the time step

  int a; //iterator variable

  array<int, 3> al{}; // Box length per processor since each subsystem is a parallel-piped
  for (a=0; a<al.size(); a++) al[a] = InitUcell[a]/cbrt(Density/4.0);

  array<int ,3> lc{}, rc{};
  /* Compute the # of cells for linked cell lists */
  for (a=0; a<3; a++) {
    lc[a] = al[a]/RCUT; 
    rc[a] = al[a]/lc[a];
  }

  if(sid == 0) {
    cout << "al = " << al[0] << al[1] << al[2] << endl;
    cout << "lc = " << lc[0] << lc[1] << lc[2] << endl;
    cout << "rc = " << rc[0] << rc[1] << rc[2] << end;
  }

  array<int, 3> nn{}, myParity{};
  
  SetTopology(nn, myParity, vid, vproc); // Uses reference to fill up nn and myparity

  // This section emulates init_conf and sets up cells in the processor
  SubSystem s;

  s.AtomCopy();
  compute_accel(s);

  cpu1 = MPI_Wtime();
  for (stepCount=1; stepCount<=StepLimit; stepCount++) {
    single_step(); 
    if (stepCount%StepAvg == 0) eval_props();
  }
  
  cpu = MPI_Wtime() - cpu1;
  if (sid == 0) printf("CPU & COMT = %le %le\n",cpu,comt);

  MPI_Finalize(); /* Clean up the MPI environment */
  return 0;
}

void SetTopology(array<int, 3> &nn, array<int, 3> &myParity, array<int, 3> vid, array<int, 3> vproc) {
  // Prepare neighbor-node ID table for a cell        
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
    for (a=0; a<3; a++) sv[ku][a] = al[a]*iv[ku][a];
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

/*--------------------------------------------------------------------*/
void single_step() {
/*----------------------------------------------------------------------
r & rv are propagated by DeltaT using the velocity-Verlet scheme.
----------------------------------------------------------------------*/
  int i,a;

  half_kick(); /* First half kick to obtain v(t+Dt/2) */
  for (i=0; i<n; i++) /* Update atomic coordinates to r(t+Dt) */
    for (a=0; a<3; a++) r[i][a] = r[i][a] + DeltaT*rv[i][a];
  atom_move();
  atom_copy();
  compute_accel(); /* Computes new accelerations, a(t+Dt) */
  half_kick(); /* Second half kick to obtain v(t+Dt) */
}

/*--------------------------------------------------------------------*/
void compute_accel(SubSystem &s) {
/*----------------------------------------------------------------------
Given atomic coordinates, r[0:n+nb-1][], for the extended (i.e., 
resident & copied) system, computes the acceleration, ra[0:n-1][], for 
the residents.
----------------------------------------------------------------------*/
  int bintra;
  double dr[3],rr,ri2,ri6,r1,rrCut,fcVal,f,vVal,lpe;

  double rr,ri2,ri6,r1;
  double Uc, Duc;

  vector<Atom>::iterator it_atom;
  
  /* Constants for potential truncation */
  rr = RCUT*RCUT; ri2 = 1.0/rr; ri6 = ri2*ri2*ri2; r1=sqrt(rr);
  Uc = 4.0*ri6*(ri6 - 1.0);
  Duc = -48.0*ri6*(ri6 - 0.5)/r1;
  
  /* Reset the potential & forces */
  for(it_atom = s.systemAtoms.begin(); it_atom != cell.atoms.end(); ++it_atom) {
    it_atom->ax = 0.0;
    it_atom->ay = 0.0;
    it_atom->az = 0.0;
  }

  /* Make a linked-cell list, lscl------------------------------------*/

  for (a=0; a<3; a++) lc2[a] = lc[a]+2;
  lcyz2 = lc2[1]*lc2[2];
  lcxyz2 = lc2[0]*lcyz2;

  /* Reset the headers, head */
  for (c=0; c<lcxyz2; c++) head[c] = EMPTY;

  /* Scan atoms to construct headers, head, & linked lists, lscl */

  for (i=0; i<n+nb; i++) {
    for (a=0; a<3; a++) mc[a] = (r[i][a]+rc[a])/rc[a];

    /* Translate the vector cell index, mc, to a scalar cell index */
    c = mc[0]*lcyz2+mc[1]*lc2[2]+mc[2];

    /* Link to the previous occupant (or EMPTY if you're the 1st) */
    lscl[i] = head[c];

    /* The last one goes to the header */
    head[c] = i;
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
    if (head[c] == -1) continue;

    /* Scan the neighbor cells (including itself) of cell c */
    for (mc1[0]=mc[0]-1; mc1[0]<=mc[0]+1; (mc1[0])++)
    for (mc1[1]=mc[1]-1; mc1[1]<=mc[1]+1; (mc1[1])++)
    for (mc1[2]=mc[2]-1; mc1[2]<=mc[2]+1; (mc1[2])++) {

      /* Calculate the scalar cell index of the neighbor cell */
      c1 = mc1[0]*lcyz2+mc1[1]*lc2[2]+mc1[2];
      /* Skip this neighbor cell if empty */
      if (head[c1] == -1) continue;

      /* Scan atom i in cell c */
      i = head[c];
      while (i != -1) {

        /* Scan atom j in cell c1 */
        j = head[c1];
        while (j != -1) {

          /* No calculation with itself */
          if (j != i) {
            /* Logical flag: intra(true)- or inter(false)-pair atom */
            bintra = (j < n);

            /* Pair vector dr = r[i] - r[j] */
            for (rr=0.0, a=0; a<3; a++) {
              dr[a] = r[i][a]-r[j][a];
              rr += dr[a]*dr[a];
            }

            /* Calculate potential & forces for intranode pairs (i < j)
               & all the internode pairs if rij < RCUT; note that for
               any copied atom, i < j */
            if (i<j && rr<rrCut) {
              ri2 = 1.0/rr; ri6 = ri2*ri2*ri2; r1 = sqrt(rr);
              fcVal = 48.0*ri2*ri6*(ri6-0.5) + Duc/r1;
              vVal = 4.0*ri6*(ri6-1.0) - Uc - Duc*(r1-RCUT);
              if (bintra) lpe += vVal; else lpe += 0.5*vVal;
              for (a=0; a<3; a++) {
                f = fcVal*dr[a];
                ra[i][a] += f;
                if (bintra) ra[j][a] -= f;
              }
            }
          } /* Endif not self */
          
          j = lscl[j];
        } /* Endwhile j not empty */

        i = lscl[i];
      } /* Endwhile i not empty */

    } /* Endfor neighbor cells, c1 */

  } /* Endfor central cell, c */

  /* Global potential energy */
  MPI_Allreduce(&lpe,&potEnergy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}

/*--------------------------------------------------------------------*/
void eval_props() {
/*----------------------------------------------------------------------
Evaluates physical properties: kinetic, potential & total energies.
----------------------------------------------------------------------*/
  double vv,lke;
  int i,a;

  /* Total kinetic energy */
  for (lke=0.0, i=0; i<n; i++) {
    for (vv=0.0, a=0; a<3; a++) vv += rv[i][a]*rv[i][a];
    lke += vv;
  }
  lke *= 0.5;
  MPI_Allreduce(&lke,&kinEnergy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  /* Energy paer atom */
  kinEnergy /= nglob;
  potEnergy /= nglob;
  totEnergy = kinEnergy + potEnergy;
  temperature = kinEnergy*2.0/3.0;

  /* Print the computed properties */
  if (sid == 0) printf("%9.6f %9.6f %9.6f %9.6f\n",
                stepCount*DeltaT,temperature,potEnergy,totEnergy);
}
