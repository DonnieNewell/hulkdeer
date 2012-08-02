#ifndef DISTRIBUTED_HOTSPOT_H
#define DISTRIBUTED_HOTSPOT_H

class SubDomain;
class Node;
void benchmarkMyself(Node*, SubDomain*, int, int, int, int, int);
void processSubDomain(int device, SubDomain *task, int timesteps,
                      float step_div_Cap, float Rx, float Ry, float Rz);
void runDistributedHotspot(int rank, int numTasks, DTYPE *data, int x_max, int y_max,
    int iterations, int pyramid_height, float step_div_Cap, float Rx, float Ry, float Rz,
        const int number_blocks_per_dimension);
void runDistributedHotspotSetData(DTYPE *MatrixPower, int num_elements);
void processSubDomain(int device, SubDomain *task, int timesteps,
                      float step_div_Cap, float Rx, float Ry, float Rz);
void receiveData(int, Node*, bool, int, float, float, float, float);
double benchmarkPCIBus(SubDomain* pS, int gpuIndex);
#endif
