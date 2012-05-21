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
void printNode(int parentRank,Node& n)
{
  for(int sd=0; sd<n.numSubDomains(); ++sd)
  {
    fprintf(stderr, "\tparent[%d] sd[%d]:%p\n",parentRank,sd,n.getSubDomain(sd));
  }
}
void printCluster(Cluster& c){
  for(int node=0; node < c.getNumNodes(); node++){
    Node &n = c.getNode(node);
    printf("node:%d weight:%f edgeWeight:%f has %d tasks.\n",node,n.getWeight(),n.getEdgeWeight(),n.numSubDomains());
//    printNode(-1,n);
    for(int child=0; child<n.getNumChildren(); ++child)
    {
      Node& ch = n.getChild(child);
      printf("\tchild:%d weight:%f edgeWeight:%f has %d tasks.\n",child,ch.getWeight(), ch.getEdgeWeight(),ch.numSubDomains());
//      printNode(n.getRank(),ch);

    }
  }
}
