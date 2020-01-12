//
// Created by Taufeq  Razakh on 1/12/20.
//
#include "pmd.hpp"

const double RCUT = 2.5; // Potential cut-off length
const double MOVED_OUT = -1.0e10;


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
        subsystem.WriteXYZ(stepCount);
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

    vector<int> head;
    //map<int, int> head;
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
    for (c=0; c<lcxyz2; c++) head.push_back(EMPTY);

    /* Scan atoms to construct headers, head, & linked lists, lscl */
    // if(subsystem.pid == 0)cout << "atoms in subsystem  = "  << subsystem.atoms.size() << endl;
    for (auto it_atom = subsystem.atoms.begin(); it_atom != subsystem.atoms.end(); ++it_atom) {
        mc[0] = (it_atom->x + rc[0]) / rc[0];
        mc[1] = (it_atom->y + rc[1]) / rc[1];
        mc[2] = (it_atom->z + rc[2]) / rc[2];
        /* Translate the vector cell index, mc, to a scalar cell index */
        c = mc[0]*lcyz2+mc[1]*lc2[2]+mc[2];

        cout.precision(6);
        cout.setf(ios::fixed, ios::floatfield);
        //if(subsystem.pid == 0) cout << "coordinates " << it_atom->x << " "  << it_atom->y << " "  << it_atom->z << endl;

        /* Link to the previous occupant (or EMPTY if you're the 1st) */
        //if(subsystem.pid == 0) cout << "before head size: " << head.size() << " c: "<< c << " i: " << i <<  " head[c]: " << head[c] << endl;
        // auto search = head.find(c);
        // if(search != head.end())
        //   lscl[distance(subsystem.atoms.begin(), it_atom)] = search->second;
        // else
        //   lscl[distance(subsystem.atoms.begin(), it_atom)] = EMPTY;
        lscl[distance(subsystem.atoms.begin(), it_atom)] = head[c];
        //if(subsystem.pid == 0) cout <<"after head[c]: " << head[c] << endl;
        //continue;
        /* The last one goes to the header */
        head[c] = distance(subsystem.atoms.begin(), it_atom);
        // if(search != head.end())
        //   head.erase(search);
        // head.insert(head.begin(), pair<int, int>(c,distance(subsystem.atoms.begin(), it_atom)));
        //if(subsystem.pid == 0) cout << "mc " << mc [0] << " " << mc [1] << " " << mc [2] << " c " << c << " lscl " << lscl[distance(subsystem.atoms.begin(), it_atom)] << " head " << head[c] << endl;
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
                //if (head.find(c) == head.end()) continue;
                if(head[c] == EMPTY) continue;

                /* Scan the neighbor cells (including itself) of cell c */
                for (mc1[0]=mc[0]-1; mc1[0]<=mc[0]+1; (mc1[0])++)
                    for (mc1[1]=mc[1]-1; mc1[1]<=mc[1]+1; (mc1[1])++)
                        for (mc1[2]=mc[2]-1; mc1[2]<=mc[2]+1; (mc1[2])++) {

                            /* Calculate the scalar cell index of the neighbor cell */
                            c1 = mc1[0]*lcyz2+mc1[1]*lc2[2]+mc1[2];
                            //if(subsystem.pid == 0) cout << "c1 = " << c1 <<endl;
                            /* Skip this neighbor cell if empty */
                            if(head[c1] == EMPTY) continue;

                            /* Scan atom i in cell c */
                            i = head[c];
                            while (i != EMPTY) {

                                /* Scan atom j in cell c1 */
                                j = head[c1];
                                while (j != EMPTY) {
                                    //if(subsystem.pid == 0)cout << "i & j :" << i << " " << j << endl;
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
                                            //if(subsystem.pid == 0) cout << " atom " << j << " ri2 :" << ri2 << " ri6 " << ri6 << " r1 " << r1 << " fcVal " << fcVal << " vVal " << vVal << " bintra " << bintra << endl;
                                            if (bintra) lpe += vVal; else lpe += 0.5*vVal;

                                            f = fcVal*dr[0];
                                            subsystem.atoms[i].ax += f;
                                            if(bintra) subsystem.atoms[j].ax -= f;
                                            //if(subsystem.pid == 0) cout << "accleration " << subsystem.atoms[j].ax << " factor " << " : " << f << endl;

                                            f = fcVal*dr[1];
                                            subsystem.atoms[i].ay += f;
                                            if(bintra) subsystem.atoms[j].ay -= f;
                                            //if(subsystem.pid == 0) cout << "accleration " << subsystem.atoms[j].ay << " factor " << " : " << f << endl;

                                            f = fcVal*dr[2];
                                            subsystem.atoms[i].az += f;
                                            if(bintra) subsystem.atoms[j].az -= f;
                                            //if(subsystem.pid == 0) cout << "accleration " << subsystem.atoms[j].az << " factor " << " : " << f << endl;
                                        }
                                    } /* Endif not self */

                                    j = lscl[j];
                                } /* Endwhile j not empty */

                                i = lscl[i];
                            } /* Endwhile i not empty */

                        } /* Endfor neighbor cells, c1 */

            } /* Endfor central cell, c */

    /* Global potential energy */
    // if(subsystem.pid == 0) cout << "local potential energy " << lpe << endl;
    MPI_Allreduce(&lpe,&subsystem.potEnergy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}

