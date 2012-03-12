#include "Decomposition.h"

Decomposition::Decomposition():domain(2){
}
Decomposition::Decomposition(const int numNodes):domain(numNodes){
}

Decomposition::Decomposition(const Decomposition &d){
  *this = d;
}
Decomposition::~Decomposition(){

}
Decomposition& Decomposition::operator=(const Decomposition& d){
  this->domain.clear();
  for ( int i=0; i<d.getNumNodes(); ++i){
    this->domain.push_back(d.getNode(i));
  }
}
Node Decomposition::getNode(const int index)const {
  Node n = this->domain.at(index);
  return n;
}

int Decomposition::getNumNodes() const {
  return domain.size();
}
void Decomposition::addNode(Node& n){
  
  domain.push_back(n);
}
void Decomposition::decompose1D(const int numElementsX){
  if(0<numElementsX){
    int numLeftX=numElementsX;
    for(int i=0; i < domain.size(); ++i){
      Node& n = domain.at(i);
      SubDomain3D s;
      s.setLength(0, n.getWeight()*numElementsX);
      s.setOffset(0, numElementsX - numLeftX);
      
      n.setSubDomain(s);

      numLeftX -= s.getLength(0);
    } //end for
  }//end if
}//end decompose1D

void Decomposition::decompose2D(const int numElementsX,const int numElementsY){
   if(0<numElementsX&&0<numElementsY){
    int numLeftX=numElementsX;
    for(int i=0; i < domain.size(); ++i){
      Node& n = domain.at(i);
      double weight = n.getWeight();
      SubDomain3D s;
      s.setLength(0, weight*numElementsX);
      s.setOffset(0, numElementsX - numLeftX);
      s.setLength(1, numElementsY);
      s.setOffset(1, 0);
      
      n.setSubDomain(s);

      numLeftX -= s.getLength(0);
    } 
  }
}//end decompose2D

void Decomposition::decompose3D(const int numElementsX,const int numElementsY,const int numElementsZ){
    if(0<numElementsX){
    int numLeftZ=numElementsZ;
    for(int i=0; i < domain.size(); ++i){
      Node& n = domain.at(i);
      SubDomain3D s;
      double weight = n.getWeight();
      s.setLength(0, numElementsX);
      s.setOffset(0, 0);
      s.setLength(1, numElementsY);
      s.setOffset(1, 0);
      s.setLength(2, weight*numElementsZ);
      s.setOffset(2, numElementsZ - numLeftZ);
      
      n.setSubDomain(s);

      numLeftZ -= s.getLength(2);
    }
  }
}//end decompose3D


void Decomposition::decompose(const int numDimensions, const int numElements[]){
  if(1 == numDimensions)
    decompose1D(numElements[0]);
  else if(2 == numDimensions)
    decompose2D(numElements[0],numElements[1]);
  else if(3 == numDimensions)
    decompose3D(numElements[0],numElements[1], numElements[2]);
}

void Decomposition::normalize(){
	double sum = 0.0;
	for(int i=0; i<domain.size(); ++i){
		sum+=domain.at(i).getWeight();
	}
	//divide each node's weight by total weight
	for(int i=0; i<domain.size(); ++i){
		domain.at(i).setWeight(domain.at(i).getWeight()/sum);
	}
}
