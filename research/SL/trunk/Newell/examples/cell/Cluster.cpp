#include "Cluster.h"

Cluster::Cluster(){

}

Cluster::~Cluster(){

}

Cluster::Cluster(int numNodes){
	nodes.resize(numNodes);
	for(int rank=0;rank<nodes.size();++rank){
		nodes.at(rank).setRank(rank);
	}
}

Node& Cluster::getNode(int index){
	return nodes.at(index);
}

int Cluster::getNumNodes(){
	return nodes.size();
}
