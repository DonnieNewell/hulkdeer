#ifndef CLUSTER_H
#define CLUSTER_H

#include "Node.h"
#include <vector>
class Cluster{
	private:
		vector<Node> nodes;
	public:
		Cluster();
		~Cluster();
		Cluster(int);
		Node& getNode(int);
		int getNumNodes();
};

void printCluster(Cluster &c);
#endif 
