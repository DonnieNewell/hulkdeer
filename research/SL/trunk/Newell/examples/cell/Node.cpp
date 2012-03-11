#include "Node.h"

Node::Node():weight(0.0){
}

Node::Node(double wt, const SubDomain3D& sd):weight(wt),subD(sd){
  
}
Node::~Node(){

}
void Node::setWeight(double newWeight){
  weight = newWeight;
}
void Node::setSubDomain(const SubDomain3D& sd){
  subD = sd;
}
double Node::getWeight(){

    return weight;
}
SubDomain3D Node::getSubDomain() const{
    return subD;
}
