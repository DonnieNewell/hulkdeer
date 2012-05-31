#ifndef CLUSTER_H
#define CLUSTER_H

#include "Node.h"
#include <vector>
class Cluster{
	private:
		vector<Node> nodes;
		vector<int> blockLocations;
	public:
		Cluster();
		~Cluster();
		Cluster(int);
                void setNumBlocks(size_t);
                size_t getNumBlocks();
                size_t getBlockLoc(size_t);
                void setBlockLoc(size_t, int);
		Node& getNode(int);
		int getNumNodes();
};

void printCluster(Cluster &c);
#endif 
