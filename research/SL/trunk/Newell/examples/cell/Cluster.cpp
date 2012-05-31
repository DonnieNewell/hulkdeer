#include "Cluster.h"
#include <cstdio>

Cluster::Cluster(){

}

Cluster::~Cluster(){

}

/* initializes the machine node objects in this cluster */
Cluster::Cluster(int numNodes){
  nodes.resize(numNodes);
  for(size_t rank=0;rank<nodes.size();++rank){
    nodes.at(rank).setRank(rank);
  }
}

/* returns the specified machine node */
Node& Cluster::getNode(int index){
  return nodes.at(index);
}

/* returns the number of machine nodes in the cluster */
int Cluster::getNumNodes(){
  return nodes.size();
}

/* sets the size of the block lookup table */
void Cluster::setNumBlocks(size_t num){
  this->blockLocations.resize(num);
}

/* returns the total number of blocks whose locations are tracked */
size_t Cluster::getNumBlocks(){
  return this->blockLocations.size();
}

/* returns the rank of the node where the block is located */
size_t Cluster::getBlockLoc(size_t index){
  return this->blockLocations.at(index);
}

/* stores the rank of the node where the specified block is located */
void Cluster::setBlockLoc(size_t index, int loc){
  this->blockLocations.at(index) = loc;
}

/* prints the structure of the cluster */
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
    printf("node:%d weight:%f edgeWeight:%e has %d tasks.\n",node,n.getWeight(),n.getEdgeWeight(),n.numSubDomains());
//    printNode(-1,n);
    for(int child=0; child<n.getNumChildren(); ++child)
    {
      Node& ch = n.getChild(child);
      printf("\tchild:%d weight:%f edgeWeight:%e has %d tasks.\n",child,ch.getWeight(), ch.getEdgeWeight(),ch.numSubDomains());
//      printNode(n.getRank(),ch);

    }
  }
}
