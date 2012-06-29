#ifndef DISTRIBUTED_CELL_H
#define DISTRIBUTED_CELL_H

class SubDomain;
class Node;
void benchmarkMyself(Node*, SubDomain*, int, int, int, int, int);
void processSubDomain(int device, SubDomain *task, int timesteps,
                      int bornMin, int bornMax, int dieMin, int dieMax);
void runDistributedCell(int rank, int numTasks, DTYPE *data, int x_max,
        int y_max, int z_max, int iterations, int bornMin, int bornMax,
        int dieMin, int dieMax);
void processSubDomain(int device, SubDomain *task, int timesteps,
                      int bornMin, int bornMax, int dieMin, int dieMax);
void receiveData(int, Node*, bool, int, int, int, int, int);
double benchmarkPCIBus(SubDomain* pS, int gpuIndex);
#endif
