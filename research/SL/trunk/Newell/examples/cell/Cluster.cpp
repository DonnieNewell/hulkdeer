#include "Cluster.h"
#include <cstdio>

Cluster::Cluster(){

}

Cluster::~Cluster(){

}

Cluster::Cluster(int numNodes){
	nodes.resize(numNodes);
	for(size_t rank=0;rank<nodes.size();++rank){
		nodes.at(rank).setRank(rank);
	}
}

Node& Cluster::getNode(int index){
	return nodes.at(index);
}

int Cluster::getNumNodes(){
	return nodes.size();
}

void printCluster(Cluster& c){
  	for(int node=0; node < c.getNumNodes(); node++){
		Node &n = c.getNode(node);
		printf("node:%d weight:%f has %d tasks.\n",node,n.getWeight(),n.numSubDomains());
		for(int child=0; child<n.getNumChildren(); ++child)
		{
			Node& ch = n.getChild(child);
				printf("\tchild:%d weight:%f has %d tasks.\n",child,ch.getWeight(),ch.numSubDomains());
			
		}
	}
}
