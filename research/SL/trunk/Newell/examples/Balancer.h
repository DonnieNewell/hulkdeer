#ifndef BALANCER_H
#define BALANCER_H

#include "Decomposition.h"
#include "Cluster.h"

class Balancer {
private:
    void balanceNode(Node&);
    void populateAdjacencies(Cluster*, int*, int*, int*, int*);

public:
    Balancer();
    ~Balancer();
    void balance(Cluster&, Decomposition&, int);
    void perfBalanceGPU(Cluster&, Decomposition&, const double);
    void perfBalance(Cluster&, Decomposition&, int);
    bool balanceNode(Node&, double, const int);
    void minCut(Cluster&);
};


#endif
