#include "Node.h"
#include <stdio.h>

Node::Node():weight(1.0) { }

Node::Node(const Node& n) {
  weight=n.weight;
  edgeWeight=n.edgeWeight;
  rank=n.rank;
  subD=n.subD;
  children = n.children;
}

Node::Node(double wt):weight(wt) { }

Node::~Node() { }

/**
 * @brief estimates how long communication and processing will take
 * @param extra additional tasks to estimate runtime for
 * @return estimated runtime given current work
 */
const double Node::getTimeEst(int extra) const {
  double est      = 0.0;
  double procTime = 1.0 / weight;
  double commTime = 1.0 / edgeWeight;
  est = (procTime + commTime) * (extra + subD.size());
  for (size_t c=0; c < children.size(); ++c) {
    const Node&  ch         = children.at(c);
    double chProcTime = 1.0 / ch.getWeight();
    double chCommTime = 1.0 / ch.getEdgeWeight();
    est += (chCommTime + commTime + chProcTime) * ch.numSubDomains();
  }
  return est;
}

Node& Node::operator=(const Node& rhs) {
  //if setting equal to itself, do nothing
  if (this != &rhs) {
    this->weight = rhs.weight;
    this->edgeWeight = rhs.edgeWeight;
    this->rank = rhs.rank;
    this->subD = rhs.subD;
    this->children = rhs.children;
  }
  return *this;
}
void Node::setRank(int newRank) {
  this->rank = newRank;
}

void Node::setNumChildren(int numChildren) {
  children.resize(numChildren);
  for (int child=0; child < numChildren; ++child) {
    children.at(child).setRank(child);
  }
}

/*
   @param runtime expected runtime
   @return task blocks needed to fill runtime for subtree
 */
int Node::getTotalWorkNeeded(const double runtime) const {
  //how many blocks could this subtree process in time
  return (int) min(this->getTotalWeight(),this->getMinEdgeWeight())*runtime;
}

/*
   @param runtime expected runtime
   @return task blocks needed to fill runtime
 */
int Node::getWorkNeeded(const double runtime) const {
  //how many blocks could this node process in time
  return (int)(runtime*weight);
}

const unsigned int Node::getNumChildren() const {
  return children.size();
}

void Node::setWeight(double newWeight) {
  weight = newWeight;
}

void Node::setEdgeWeight(double newEdgeWeight) {
  edgeWeight = newEdgeWeight;
}

void Node::addSubDomain(SubDomain* sd) {
  subD.push_back(sd);
}

const int Node::getRank() const {
  return this->rank;
}

const double Node::getMinEdgeWeight() const {
  double minWeight = edgeWeight;
  for (size_t child=0; child < children.size(); ++child) {
    minWeight = min(children.at(child).getEdgeWeight(), minWeight);
  }

#ifdef DEBUG
  fprintf(stderr, "node[%d] edgeWeight:%f min edge Weight:%f.\n",
          rank, edgeWeight, minWeight);
#endif
  return minWeight;
}

const double Node::getTotalWeight() const {
  double total = weight;
  for (size_t child=0; child < children.size(); ++child) {
    total += children.at(child).getWeight();
  }

#ifdef DEBUG
  fprintf(stderr, "node[%d] weight:%f total_weight:%f.\n",rank,weight,total);
#endif
  return total;
}

const double Node::getEdgeWeight() const {
  return edgeWeight;
}

const double Node::getWeight() const {
  return weight;
}

const Node& Node::getChild(int index) const {
  return children.at(index);
}

Node& Node::getChild(int index) {
  const Node& node = static_cast<const Node &>( *this );
  return const_cast<Node &>( node.getChild(index) );
}

SubDomain* Node::getSubDomain(int index) const {
  return subD.at(index);
}

SubDomain* Node::globalGetSubDomain(int index) const {
  unsigned int currentIndex = static_cast<unsigned int>(index);
  if (this->numSubDomains() > currentIndex) {
    return this->getSubDomain(currentIndex);
  } else {
    currentIndex -= this->numSubDomains();
    for (unsigned int gpuIndex = 0;
        gpuIndex < this->getNumChildren();
        ++gpuIndex) {
      const Node& kGpu = this->getChild(gpuIndex);
      if (kGpu.numSubDomains() > currentIndex) {
        return kGpu.getSubDomain(currentIndex);
      }
      currentIndex -= kGpu.numSubDomains();
    }
  }
  return NULL;
}

SubDomain* Node::getSubDomainLinear(int index) const {
  for (unsigned int blockIndex = 0; blockIndex < numTotalSubDomains(); ++blockIndex) {
    SubDomain* block = globalGetSubDomain(blockIndex);
    if (block->getLinIndex() == index) {
      return block;
    }
  }
#ifdef DEBUG
  fprintf(stderr, "\n[%d]couldn't find block with linear index:%d", this->getRank(), index);
#endif
  return NULL;
}

SubDomain* Node::popSubDomain() {
  SubDomain* s = subD.back();
  subD.pop_back();
  return s;
}

const unsigned int Node::numSubDomains() const {
  return subD.size();
}

const unsigned int Node::numTotalSubDomains() const {
  unsigned int total = subD.size();
  for (size_t child=0; child < children.size(); ++child) {
    total += children.at(child).numSubDomains();
  }
  return total;
}

void printNode(Node& node) {
  printf("**********************************************\n");
  printf("node[%d]:\n", node.getRank());
  for (unsigned int i = 0; i < node.numTotalSubDomains(); ++i) {
    SubDomain* sub_domain = node.globalGetSubDomain(i);
    printf("  block[%d]\n", sub_domain->getLinIndex());
    printSubDomain(sub_domain);
  }
}
