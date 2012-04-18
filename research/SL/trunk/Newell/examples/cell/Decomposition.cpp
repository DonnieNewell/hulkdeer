#include "Decomposition.h"
#include <stdio.h>
Decomposition::Decomposition(){
}
Decomposition::Decomposition(const int numSubDomains){
}

Decomposition::Decomposition(const Decomposition &d){
  *this = d;
}
Decomposition::~Decomposition(){

}
Decomposition& Decomposition::operator=(const Decomposition& d){
  this->domain.clear();
  for ( int i=0; i<d.getNumSubDomains(); ++i){
    this->domain.push_back(d.getSubDomain(i));
  }
}
SubDomain3D Decomposition::getSubDomain(const int index)const {
  SubDomain3D s = this->domain.at(index);
  return s;
}

int Decomposition::getNumSubDomains() const {
  return domain.size();
}
void Decomposition::addSubDomain(SubDomain3D& s){
  
  domain.push_back(s);
}
void Decomposition::decompose1D(const int numElementsX){
  if(0<numElementsX){
    int numLeftX=numElementsX;
    const int num_chunks = 64;
    int stride = numElementsX/num_chunks;

    for(int i=0; i < domain.size(); ++i){
      SubDomain3D& s = domain.at(i);
      
      
      s.setLength(0, stride);
      s.setOffset(0, numElementsX - numLeftX);
      
      numLeftX -= s.getLength(0);
    } //end for
  }//end if
}//end decompose1D

void Decomposition::decompose2D(const int numElementsX,const int numElementsY){
   if(0<numElementsX&&0<numElementsY){
    int numLeftX=numElementsX;
    int numLeftY=numElementsY;
    const int num_chunks = 8;
    int x_chunk_width = numElementsX/num_chunks;
    int y_chunk_width = numElementsY/num_chunks;

    for(int i=0; i < domain.size(); ++i){
      SubDomain3D& s = domain.at(i);
      s.setLength(0, x_chunk_width);
      s.setOffset(0, numElementsX - numLeftX);
      s.setLength(1, y_chunk_width);
      s.setOffset(1, numElementsY - numLeftY);
      
      numLeftX -= s.getLength(0);
      numLeftY -= s.getLength(1);
    } 
  }
}//end decompose2D

void Decomposition::decompose3D(const int numElementsX,const int numElementsY,const int numElementsZ){
  fprintf(stdout, "entering decompose3D\n");
  int width = 4;  
  int blockDimX = static_cast<int>((numElementsX/(double)width)+.5);
  int blockDimY =  static_cast<int>((numElementsY/(double)width)+.5);
  int blockDimZ =  static_cast<int>((numElementsZ/(double)width)+.5);
  SubDomain3D s;
  fprintf(stdout, "clearing subd vector decompose3D\n");
  domain.clear();
  for(int i=0; i < width; ++i){
    for(int j=0; j < width; ++j){
      for(int k=0; k < width; ++k){
      
        s.setLength(0, blockDimX);
        s.setOffset(0, blockDimX*i);
        s.setLength(1, blockDimY);
        s.setOffset(1, blockDimY*j);
        s.setLength(2, blockDimZ);
        s.setOffset(2, blockDimZ*k);
      
	domain.push_back(s);
      }
    }
  }
 fprintf(stdout,"domain.size():%zu\n",domain.size()); 

}//end decompose3D


void Decomposition::decompose(const int numDimensions, const int numElements[3]){
fprintf(stderr,"decompose(%d dimensions, [%d][%d][%d]\n",numDimensions,numElements[0],numElements[1],numElements[2]);
  if(1 == numDimensions)
    decompose1D(numElements[0]);
  else if(2 == numDimensions)
    decompose2D(numElements[0],numElements[1]);
  else if(3 == numDimensions)
    decompose3D(numElements[0],numElements[1], numElements[2]);
}

