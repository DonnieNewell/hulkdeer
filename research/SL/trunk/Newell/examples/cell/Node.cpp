#include "Node.h"
#include <stdio.h>

Node::Node():weight(1.0){
}
Node::Node(const Node& n){
  weight=n.weight;
  edgeWeight=n.edgeWeight;
  rank=n.rank;
  subD=n.subD;
  children = n.children;
}

Node::Node(double wt):weight(wt){

}
Node::~Node(){

}

/**
 * @brief estimates how long communication and processing will take
 * @param extra additional tasks to estimate runtime for
 * @return estimated runtime given current work
 */ 
const double Node::getTimeEst(int extra) const
{
  double est      = 0.0             ;
  double procTime = 1.0 / weight    ;
  double commTime = 1.0 / edgeWeight;
  
  est = (procTime + commTime) * (extra + subD.size());
  
  for(size_t c=0; c < children.size(); ++c)
  {
    const Node&  ch         = children.at(c)          ;
    double chProcTime = 1.0 / ch.getWeight()    ;
    double chCommTime = 1.0 / ch.getEdgeWeight();
    est += (chCommTime + commTime + chProcTime) * ch.numSubDomains();
  }

  return est;
}

Node& Node::operator=(const Node& rhs){
  //if setting equal to itself, do nothing
  if(this != &rhs){
    this->weight = rhs.weight;
    this->edgeWeight = rhs.edgeWeight;
    this->rank = rhs.rank;
    this->subD = rhs.subD;
    this->children = rhs.children;
  }
  return *this;
}
void Node::setRank(int newRank){
  this->rank = newRank;
}
void Node::setNumChildren(int numChildren){
  children.resize(numChildren);
  for(int child=0; child<numChildren; ++child)
  {
    children.at(child).setRank(child);
  }
}

/*
   @param runtime expected runtime
   @return task blocks needed to fill runtime for subtree
 */    
int Node::getTotalWorkNeeded(const double runtime) const
{
  //how many blocks could this subtree process in time
  return (int) min(this->getTotalWeight(),this->getMinEdgeWeight())*runtime;
}

/*
   @param runtime expected runtime
   @return task blocks needed to fill runtime
 */    
int Node::getWorkNeeded(const double runtime) const
{
  //how many blocks could this node process in time
  return (int)(runtime*weight);
}

const int Node::getNumChildren() const{
  return children.size();
}
void Node::setWeight(double newWeight){
  weight = newWeight;
}
void Node::setEdgeWeight(double newEdgeWeight){
  edgeWeight = newEdgeWeight;
}
void Node::addSubDomain(SubDomain3D* sd){
  subD.push_back(sd);
}
const int Node::getRank() const{

  return this->rank;
}
const double Node::getMinEdgeWeight() const
{
  double minWeight = edgeWeight;
  for(size_t child=0; child<children.size(); ++child)
  {
    minWeight = min(children.at(child).getEdgeWeight(), minWeight);
  }

#ifdef DEBUG
  fprintf(stderr, "node[%d] edgeWeight:%f min edge Weight:%f.\n",rank,edgeWeight,minWeight);
#endif
  return minWeight;
}

const double Node::getTotalWeight() const{
  double total = weight;
  for(size_t child=0; child<children.size(); ++child)
  {
    total += children.at(child).getWeight();
  }

#ifdef DEBUG
  fprintf(stderr, "node[%d] weight:%f total_weight:%f.\n",rank,weight,total);
#endif
  return total;
}
const double Node::getEdgeWeight() const{

  return edgeWeight;
}
const double Node::getWeight() const{

  return weight;
}
Node& Node::getChild(int index) {
  return children.at(index);
}
SubDomain3D* Node::getSubDomain(int index) {
  return subD.at(index);
}

SubDomain3D* Node::popSubDomain() 
{
  SubDomain3D* s = subD.back();
  subD.pop_back();
  return s;
}	
const int Node::numSubDomains() const{
  return subD.size();
}

const int Node::numTotalSubDomains() const{
  int total = subD.size();

  for(size_t child=0; child<children.size(); ++child)
  {
    total += children.at(child).numSubDomains();
  }

  return total;
}
