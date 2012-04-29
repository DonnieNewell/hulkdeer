#include "Balancer.h"
#include <cmath>
#include <cstdio>

Balancer::Balancer(){
}

Balancer::~Balancer(){

}

void Balancer::balance(Cluster &cluster, Decomposition& decomp){
#ifdef DEBUG
	fprintf(stderr, "in balance();\n");
#endif
	int subd_per_node = ceil(decomp.getNumSubDomains()/(float)cluster.getNumNodes());
	for(int node = 0; node < cluster.getNumNodes(); ++node){
#ifdef DEBUG
		fprintf(stderr, "node:%d\n",node);
#endif
		Node& n = cluster.getNode(node);
		for(int subd=0; subd<subd_per_node; ++subd){
			int index = node*subd_per_node + subd;
			SubDomain3D &s = decomp.getSubDomain(index);
			n.addSubDomain(s);
		}
	}
#ifdef DEBUG
	fprintf(stderr, "leaving balance();\n");
#endif
}
