#include "Node.h"

Node::Node():weight(1.0){
}
Node::Node(const Node& n){
	weight=n.weight;
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
}
const int Node::getNumChildren() const{
  return children.size();
}
void Node::setWeight(double newWeight){
  weight = newWeight;
}
void Node::addSubDomain(const SubDomain3D& sd){
  subD.push_back(sd);
}
const int Node::getRank() const{

    return this->rank;
}
const double Node::getWeight() const{

    return weight;
}
Node& Node::getChild(int index) {
    return children.at(index);
}
SubDomain3D& Node::getSubDomain(int index) {
    return subD.at(index);
}

const int Node::numSubDomains() const{
    return subD.size();
}
