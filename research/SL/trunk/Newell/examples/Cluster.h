#ifndef CLUSTER_H
#define CLUSTER_H

#include "Node.h"
#include <vector>
class Cluster{
  public:
    Cluster();
    ~Cluster();
    Cluster(int);
    void setNumBlocks(size_t);
    size_t getNumBlocks();
    size_t getBlockLoc(size_t);
    void setBlockLoc(size_t, int);
    void storeBlockLocs();
    Node& getNode(int);
    unsigned int getNumNodes() const;
    SubDomain* getBlockLinear(const int);
  private:
    void updateBlockNeighbors();
    vector<Node> nodes;
    vector<int> blockLocations;
};

void printCluster(Cluster &c);
void printBlockLocations(Cluster &c);
#endif
