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
  return (int) (this->getTotalWeight()*runtime);
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
