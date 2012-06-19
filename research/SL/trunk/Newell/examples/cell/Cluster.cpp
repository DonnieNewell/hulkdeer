#include "Cluster.h"
#include <cstdio>

Cluster::Cluster() { }

Cluster::~Cluster() { }

/* initializes the machine node objects in this cluster */
Cluster::Cluster(int numNodes) {
  nodes.resize(numNodes);
  for (size_t rank=0;rank<nodes.size();++rank) {
    nodes.at(rank).setRank(rank);
  }
}

/* returns the specified machine node */
Node& Cluster::getNode(int index) {
  return nodes.at(index);
}

/* returns the number of machine nodes in the cluster */
unsigned int Cluster::getNumNodes() const {
  return nodes.size();
}

/* sets the size of the block lookup table */
void Cluster::setNumBlocks(size_t num) {
  this->blockLocations.resize(num);
}

/* returns the total number of blocks whose locations are tracked */
size_t Cluster::getNumBlocks() {
  return this->blockLocations.size();
}

/* returns the rank of the node where the block is located */
size_t Cluster::getBlockLoc(size_t index) {
  return this->blockLocations.at(index);
}

/* sets the rank of each neighbor for each subdomain */
void Cluster::updateBlockNeighbors () {
  /* loop through every physical node */
  for (unsigned int rank=0; rank < this->getNumNodes(); ++rank) {
    const Node& n = this->getNode(rank);
    /* loop through work on each node */
    for (unsigned int block = 0; block < n.numTotalSubDomains(); ++block) {
      SubDomain3D *currentBlock = n.globalGetSubDomain(block);
      currentBlock->setNeighbors(this->blockLocations);
//      printNeighbors(currentBlock);  // DEBUG
    }
    printf("node[%d] has %d total dataBlocks.\n", rank, n.numTotalSubDomains());
  }
}

/* stores the locations of all current work in the cluster */
void Cluster::storeBlockLocs() {
  /* loop through every physical node */
  for (unsigned int rank=0; rank < this->getNumNodes(); ++rank) {
    const Node& n = this->getNode(rank);

    /* loop through work on each node */
    for (unsigned int block = 0; block < n.numSubDomains(); ++block) {
      SubDomain3D *currentBlock = n.getSubDomain(block);
      int linIndex = currentBlock->getLinIndex();
      this->setBlockLoc(linIndex, rank);
    }

    /* loop through all devices on node */
    for (unsigned int c = 0; c < n.getNumChildren(); ++c) {
      const Node& child = n.getChild(c);

      /* loop through all work on each device */
      for(unsigned int block = 0; block < child.numSubDomains(); ++block) {
        SubDomain3D *currentBlock = child.getSubDomain(block);
        int linIndex = currentBlock->getLinIndex();
        this->setBlockLoc(linIndex, rank);
      }
    }
  }
  /* stores the ranks of all neighbors on each task block */
  this->updateBlockNeighbors();
}

/* stores the rank of the node where the specified block is located */
void Cluster::setBlockLoc(size_t index, int loc) {
  this->blockLocations.at(index) = loc;
}

/* prints the structure of the cluster */
void printNode(int parentRank,Node& n) {
  for (unsigned int sd=0; sd < n.numSubDomains(); ++sd) {
    fprintf(stderr, "\tparent[%d] sd[%d]:%p\n",
            parentRank, sd, n.getSubDomain(sd));
  }
}

void printCluster(Cluster& c) {
  for (unsigned int node=0; node < c.getNumNodes(); node++) {
    Node &n = c.getNode(node);
    printf("node:%d weight:%f edgeWeight:%e has %d tasks.\n", node,
          n.getWeight(), n.getEdgeWeight(), n.numSubDomains());
    for (unsigned int child=0; child < n.getNumChildren(); ++child) {
      Node& ch = n.getChild(child);
      printf("\tchild:%d weight:%f edgeWeight:%e has %d tasks.\n",
            child, ch.getWeight(), ch.getEdgeWeight(), ch.numSubDomains());
    }
  }
}

void printBlockLocations(Cluster& c) {
  for (unsigned int block=0; block < c.getNumBlocks(); ++block) {
    printf("block[%d] on node[%d]\n", block, c.getBlockLoc(block));
  }
}
