#ifndef DISTRIBUTED_CELL_H
#define DISTRIBUTED_CELL_H

class SubDomain;
class Node;

/** process the data and store the metrics on the Node for later use */
void benchmarkMyself(Node*, SubDomain*, int, const int, int, int, int, int);

/** main function for processing large block of data on a cluster with CUDA and MPI. */
void runDistributedCell(const int, const int, DTYPE *, const int, const int,
        const int, const int, const int, const int, const int, const int,
        const int, const int, const bool, const int);

/** performs stencil calculation on the specified device using the specified 
  pyramid height */
void processSubDomain(int device, SubDomain *task, const int kPyramidHeight,
        int bornMin, int bornMax, int dieMin, int dieMax);

/** processes the outer halo, or ghost zone, of the data */
void processSubDomainOuter(int device, SubDomain *task, const int kPyramidHeight,
        int bornMin, int bornMax, int dieMin, int dieMax);

/** processes the inner data. Useful so we can asynchronously communicate the
  ghost zones */
void processSubDomainInner(int device, SubDomain *task, const int kPyramidHeight,
        int bornMin, int bornMax, int dieMin, int dieMax);

/** reads in data sent to the node over MPI */
void receiveData(int, Node*, bool, int, int, int, int, int);

/** measure the latency of sending data to the GPU. */
double benchmarkPCIBus(SubDomain* pS, int gpuIndex);

#endif
