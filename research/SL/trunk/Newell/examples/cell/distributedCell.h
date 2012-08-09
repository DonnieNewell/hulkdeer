#ifndef DISTRIBUTED_CELL_H
#define DISTRIBUTED_CELL_H

class SubDomain;
class Node;
void benchmarkMyself(Node*, SubDomain*, int, const int, int, int, int, int);
void runDistributedCell(const int, const int, DTYPE *, const int, const int,
        const int, const int, const int, const int, const int, const int,
        const int, const int, const bool, const int);
void processSubDomain(int device, SubDomain *task, const int kPyramidHeight,
        int bornMin, int bornMax, int dieMin, int dieMax);
void receiveData(int, Node*, bool, int, int, int, int, int);
double benchmarkPCIBus(SubDomain* pS, int gpuIndex);

#endif
