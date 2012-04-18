#include "Node.h"

Node::Node():weight(1.0){
}

Node::Node(double wt):weight(wt){
  
}
Node::~Node(){

}
void Node::setWeight(double newWeight){
  weight = newWeight;
}
void Node::addSubDomain(const SubDomain3D& sd){
  subD.push_back(sd);
}
const double Node::getWeight() const{

    return weight;
}
SubDomain3D Node::getSubDomain(int index) const{
    return subD.at(index);
}

const int Node::numSubDomains() const{
    return subD.size();
}
