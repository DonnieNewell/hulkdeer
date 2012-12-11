#ifndef DISTRIBUTED_HOTSPOT_H
#define DISTRIBUTED_HOTSPOT_H

class SubDomain;
class Node;
void benchmarkMyself(Node*, SubDomain*, int, const int, float, float, float, float);
void processSubDomain(int device, SubDomain *task, int timesteps,
                      float step_div_Cap, float Rx, float Ry, float Rz);
void runDistributedHotspot(const int, const int, DTYPE*, const int, const int,
        const int, const int, const float, const float, const float, const float,
        const int, const bool, const int);
void runDistributedHotspotSetData(DTYPE *MatrixPower, int num_elements);
void processSubDomain(int device, SubDomain *task, int timesteps,
                      float step_div_Cap, float Rx, float Ry, float Rz);
void receiveData(int, Node*, bool, int, float, float, float, float);
double benchmarkPCIBus(SubDomain* pS, int gpuIndex);
#endif
