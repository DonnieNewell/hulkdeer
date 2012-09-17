#include "Cluster.h"
#include <cstdio>

Cluster::Cluster() {
}

Cluster::~Cluster() {
}

/* initializes the machine node objects in this cluster */
Cluster::Cluster(int numNodes) {
    nodes.resize(numNodes);
    for (size_t rank = 0; rank < nodes.size(); ++rank) {
        nodes.at(rank).setRank(rank);
    }
}

unsigned int Cluster::getNumTotalGPUs() {
    unsigned int total_gpus = 0;
    for (unsigned int node_index = 0;
            node_index < this->getNumNodes();
            ++node_index) {
        Node& node = this->getNode(node_index);
        total_gpus += node.getNumChildren();
    }
    return total_gpus;
}

Node& Cluster::getGlobalGPU(const int gpu_index) {
    int current_index = gpu_index;
    const int kExplodeIndex = -1;
    const int kNumNodes = static_cast<int>(this->getNumNodes());
    for (int cpu_index = 0; cpu_index < kNumNodes; ++cpu_index) {
        Node& current_node = this->getNode(cpu_index);
        const int kNumChildren = static_cast<int>(current_node.getNumChildren());
        if (current_index < kNumChildren)
            return current_node.getChild(current_index);
        current_index -= current_node.getNumChildren();
    }
    // hack and I hate this so much but no time
    return this->getNode(kExplodeIndex);
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

void Cluster::distributeBlocks(Decomposition* blocks) {
  const unsigned int kNumBlocks = blocks->getNumSubDomains();
  const unsigned int kNumNodes = getNumNodes();
  for (unsigned int node_index = 0; node_index < kNumNodes; ++node_index) {
    Node& node = getNode(static_cast<int>(node_index));

    /* give node all blocks */
    int blocks_needed = node.getBalCount();
    for (int block_id = 0; block_id < blocks_needed; ++block_id)
      node.addSubDomain(blocks->popSubDomain());

    /* give blocks to node's children */
    const int kNumChildren = node.getNumChildren();
    for (int child_id = 0; child_id < kNumChildren; ++child_id) {
      Node& child = node.getChild(child_id);
      blocks_needed = child.getBalCount();
      for (int block_id = 0; block_id < blocks_needed; ++block_id)
        child.addSubDomain(blocks->popSubDomain());
    }
  }
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
void Cluster::updateBlockNeighbors() {
    /* loop through every physical node */
    for (unsigned int rank = 0; rank < this->getNumNodes(); ++rank) {
        const Node& n = this->getNode(rank);
        /* loop through work on each node */
        for (unsigned int block = 0; block < n.numTotalSubDomains(); ++block) {
            SubDomain *currentBlock = n.globalGetSubDomain(block);
            currentBlock->setNeighbors(this->blockLocations);
            //      printNeighbors(currentBlock);  // DEBUG
        }
        // printf("node[%d] has %d total dataBlocks.\n", rank, n.numTotalSubDomains());
    }
}

/* stores the locations of all current work in the cluster */
void Cluster::storeBlockLocs() {
    /* loop through every physical node */
    for (unsigned int rank = 0; rank < this->getNumNodes(); ++rank) {
        const Node& n = this->getNode(rank);

        /* loop through work on each node */
        for (unsigned int block = 0; block < n.numSubDomains(); ++block) {
            SubDomain *currentBlock = n.getSubDomain(block);
            int linIndex = currentBlock->getLinIndex();
            //fprintf(stderr, "***linIndex:%d.\n", linIndex);
            this->setBlockLoc(linIndex, rank);
        }

        /* loop through all devices on node */
        for (unsigned int c = 0; c < n.getNumChildren(); ++c) {
            const Node& child = n.getChild(c);
            /* loop through all work on each device */
            for (unsigned int block = 0; block < child.numSubDomains(); ++block) {
                SubDomain *currentBlock = child.getSubDomain(block);
                int linIndex = currentBlock->getLinIndex();
                //fprintf(stderr, "***linIndex:%d.", linIndex);
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

SubDomain* Cluster::getBlockLinear(const int kLinearIndex) {
    SubDomain* result = NULL;
    for (unsigned int node_index = 0; node_index < this->getNumNodes(); node_index++) {
        Node &node = this->getNode(node_index);
        result = node.getSubDomainLinear(kLinearIndex);
        if (NULL != result) return result;
        for (unsigned int child = 0; child < node.getNumChildren(); ++child) {
            Node& child_node = node.getChild(child);
            result = child_node.getSubDomainLinear(kLinearIndex);
            if (NULL != result) return result;
        }
    }
    return result;
}

/* prints the structure of the cluster */
void printNode(int parentRank, Node& n) {
    for (unsigned int sd = 0; sd < n.numSubDomains(); ++sd) {
        fprintf(stderr, "\tparent[%d] sd[%d]:%p\n",
                parentRank, sd, n.getSubDomain(sd));
    }
}

void printCluster(Cluster& c) {
    printf("There are %d GPUS on this cluster.\n", c.getNumTotalGPUs());
    for (unsigned int node = 0; node < c.getNumNodes(); node++) {
        Node &n = c.getNode(node);
        printf(
            "node:%d compute:%7.2e comm:%7.2e tasks:%d external_neighbors:%d\n",
                node, n.getWeight(), n.getEdgeWeight(), n.numSubDomains(),
                n.numExternalBlockNeighbors());
        for (unsigned int child = 0; child < n.getNumChildren(); ++child) {
            Node& ch = n.getChild(child);
            printf("\tchild:%d compute:%7.2e comm:%7.2e tasks:%d \
                        external_neighbors:%d\n", child, ch.getWeight(),
                    ch.getEdgeWeight(), ch.numSubDomains(),
                    ch.numExternalBlockNeighbors());
        }
    }
}

void printBlockLocations(Cluster& c) {
    for (unsigned int block = 0; block < c.getNumBlocks(); ++block) {
        printf("block[%d] on node[%zu]\n", block, c.getBlockLoc(block));
    }
}
