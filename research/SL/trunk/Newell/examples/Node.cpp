#include "Node.h"
#include <stdio.h>

using namespace std;

Node::Node() : weight(1.0), is_CPU(true) {
}

Node::Node(const Node& source_node) {
  is_CPU = source_node.is_CPU;
  weight = source_node.weight;
  edgeWeight = source_node.edgeWeight;
  rank = source_node.rank;
  subD = source_node.subD;
  children = source_node.children;
  linear_lookup = source_node.linear_lookup;
}

Node::Node(double wt) : weight(wt), is_CPU(true) {
}

Node::~Node() {
}

/**
 * @brief estimates how long communication and processing will take
 * @param extra additional tasks to estimate runtime for
 * @return estimated runtime given current work
 */
double Node::getTimeEst(const int kExtraBlocks, const int kConfig) const {
  double processing_time = 1.0 / this->getTotalWeight(kConfig);
  double communication_time = 1.0 / this->getMinEdgeWeight(kConfig);
  double time_per_block = processing_time + communication_time;
  double blocks_to_process = this->numTotalSubDomains() + kExtraBlocks;
  return blocks_to_process * time_per_block;
}

Node& Node::operator=(const Node& rhs) {
  //if setting equal to itself, do nothing
  if (this != &rhs) {
    this->weight = rhs.weight;
    this->edgeWeight = rhs.edgeWeight;
    this->rank = rhs.rank;
    this->subD = rhs.subD;
    this->children = rhs.children;
    this->linear_lookup = rhs.linear_lookup;
  }
  return *this;
}

void Node::setRank(int newRank) {
  this->rank = newRank;
}

void Node::setNumChildren(const int kNumChildren) {
  children.resize(kNumChildren);
  for (int child_index = 0; child_index < kNumChildren; ++child_index) {
    Node& gpu = children.at(child_index);
    gpu.setRank(child_index);
    gpu.setCPU(false);
  }
}

void Node::setCPU(const bool kIsCPU) {
  this->is_CPU = kIsCPU;
}

bool Node::isCPU() const {
  return this->is_CPU;
}

/*
   @param runtime expected runtime
   @return task blocks needed to fill runtime for subtree
 */
int Node::getTotalWorkNeeded(const double kRuntime, const int kConfig) const {
  //how many blocks could this subtree process in time
  double process_rate = 1.0 / getTotalWeight(kConfig);
  double communication_rate = 1.0 / getMinEdgeWeight(kConfig);
  double seconds_per_block = process_rate + communication_rate;
  return static_cast<int> (kRuntime / seconds_per_block);
}

/*
   @param runtime expected runtime
   @return task blocks needed to fill runtime
 */
int Node::getWorkNeeded(const double kRuntime) const {
  //how many blocks could this node process in time
  return static_cast<int> (kRuntime * weight);
}

unsigned int Node::getNumChildren() const {
  return children.size();
}

void Node::setWeight(double new_weight) {
  weight = new_weight;
}

void Node::setEdgeWeight(double new_edge_weight) {
  edgeWeight = new_edge_weight;
}

void Node::addSubDomain(SubDomain* new_sub_domain) {
  subD.push_back(new_sub_domain);
  linear_lookup[new_sub_domain->getLinIndex()] = subD.size() - 1;
  //printf("[%d] subdomain: %d block index: %d\n", this->getRank(), new_sub_domain->getLinIndex(), subD.size() - 1);
}

int Node::getRank() const {
  return this->rank;
}

double Node::getMinEdgeWeight(const int config) const {
  const int kCPUAndGPU = 0;
  const int kCPUOnly = 1;
  const int kGPUOnly = 2;
  double minimum_weight = edgeWeight;
  if (kCPUAndGPU == config || kGPUOnly == config) {
    for (size_t child_index = 0; child_index < children.size(); ++child_index) {
      double child_weight = children.at(child_index).getEdgeWeight();
      minimum_weight = min(child_weight, minimum_weight);
    }
  }

#ifdef DEBUG
  fprintf(stderr, "node[%d] edgeWeight:%f min edge Weight:%f.\n",
          rank, edgeWeight, minimum_weight);
#endif
  return minimum_weight;
}

double Node::getTotalWeight(const int config) const {
  const int kCPUAndGPU = 0;
  const int kCPUOnly = 1;
  const int kGPUOnly = 2;
  double total = 0.0;
  double gpu_total = 0.0, cpu_total = 0.0;
  if (isCPU())
    cpu_total += getWeight();
  else
    gpu_total += getWeight();

  for (size_t child_index = 0; child_index < children.size(); ++child_index) {
    if (children.at(child_index).isCPU())
      cpu_total += children.at(child_index).getWeight();
    else
      gpu_total += children.at(child_index).getWeight();
  }

  total = cpu_total + gpu_total;
  if (kCPUOnly == config)
    total -= gpu_total;
  else if (kGPUOnly == config)
    total -= cpu_total;

#ifdef DEBUG
  fprintf(stderr, "node[%d] weight:%f total_weight:%f.\n", rank, weight, total);
#endif
  return total;
}

double Node::getEdgeWeight() const {
  return edgeWeight;
}

double Node::getWeight() const {
  return weight;
}

const Node& Node::getChild(int index) const {
  return children.at(index);
}

Node& Node::getChild(int index) {
  const Node& node = static_cast<const Node &> (*this);
  return const_cast<Node &> (node.getChild(index));
}

SubDomain* Node::getSubDomain(int index) const {
  return subD.at(index);
}

SubDomain* Node::globalGetSubDomain(int index) const {
  unsigned int current_index = static_cast<unsigned int> (index);
  if (this->numSubDomains() > current_index) {
    return this->getSubDomain(current_index);
  } else {
    current_index -= this->numSubDomains();
    for (unsigned int gpu_index = 0;
            gpu_index < this->getNumChildren();
            ++gpu_index) {
      const Node& kGpu = this->getChild(gpu_index);
      if (kGpu.numSubDomains() > current_index) {
        return kGpu.getSubDomain(current_index);
      }
      current_index -= kGpu.numSubDomains();
    }
  }
  return NULL;
}

SubDomain* Node::getSubDomainLinear(int index) const {
  map<int, int>::const_iterator it;
  it = this->linear_lookup.find(index);
  if (!(it == linear_lookup.end())) {
    return subD.at(it->second);
  }
  for (unsigned int gpu_index = 0; gpu_index < getNumChildren(); ++gpu_index) {
    const Node& gpu = children.at(gpu_index);
    it = gpu.linear_lookup.find(index);
    if (!(it == linear_lookup.end())) {
        return gpu.subD.at(it->second);
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
  int linear_index = s->getLinIndex();
  linear_lookup.erase(linear_lookup.find(linear_index));
  return s;
}

unsigned int Node::numSubDomains() const {
  return subD.size();
}

unsigned int Node::numTotalSubDomains() const {
  unsigned int total = subD.size();
  const int kNumChildren = children.size();
  for (size_t child = 0; child < kNumChildren; ++child) {

    total += children.at(child).numSubDomains();
  }
  return total;
}

void printNode(Node& node) {
  printf("**********************************************\n");
  printf("node[%d]:\n", node.getRank());
  for (unsigned int block_index = 0;
          block_index < node.numTotalSubDomains();
          ++block_index) {
    SubDomain* sub_domain = node.globalGetSubDomain(block_index);
    printf("  block[%d]\n", sub_domain->getLinIndex());
    printSubDomain(sub_domain);
  }
}
