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
return *this;
}
const SubDomain3D& Decomposition::getSubDomain(const int index) const{
  return this->domain.at(index);
}
SubDomain3D& Decomposition::getSubDomain(const int index){
  return this->domain.at(index);
}

int Decomposition::getNumSubDomains() const {
  return domain.size();
}
void Decomposition::addSubDomain(SubDomain3D& s){
  
  domain.push_back(s);
}
void Decomposition::decompose1D(int* buffer, const int numElementsX){
  if(0<numElementsX){
    int numLeftX=numElementsX;
    const int num_chunks = 64;
    int stride = numElementsX/num_chunks;

    for(size_t i=0; i < domain.size(); ++i){
      SubDomain3D& s = domain.at(i);
      
      
      s.setLength(0, stride);
      s.setOffset(0, numElementsX - numLeftX);
      
      numLeftX -= s.getLength(0);
    } //end for
  }//end if
}//end decompose1D

void Decomposition::decompose2D(int* buffer, const int numElementsRows,const int numElementsCols){
   if(0<numElementsRows&&0<numElementsCols){
    int numLeftX=numElementsRows;
    int numLeftY=numElementsCols;
    const int num_chunks = 8;
    int x_chunk_width = numElementsRows/num_chunks;
    int y_chunk_width = numElementsCols/num_chunks;

    for(size_t i=0; i < domain.size(); ++i){
      SubDomain3D& s = domain.at(i);
      s.setLength(0, x_chunk_width);
      s.setOffset(0, numElementsRows - numLeftX);
      s.setLength(1, y_chunk_width);
      s.setOffset(1, numElementsCols - numLeftY);
      
      numLeftX -= s.getLength(0);
      numLeftY -= s.getLength(1);
    } 
  }
}//end decompose2D


void Decomposition::copyBlock(int* buffer, SubDomain3D& s, const int numElementsDepth,const int numElementsRows,const int numElementsCols){
    int* newBuff = NULL;
    int totalSize = s.getLength(0)*s.getLength(1)*s.getLength(2);
    newBuff = new int[totalSize];
    
    //stage sub-domain data into contiguous memory
    for(int depth = 0; depth < s.getLength(0); depth++){
        for(int row = 0; row < s.getLength(1); row++){
            for(int col = 0; col < s.getLength(2); col++){
                int newIndex = depth*s.getLength(1)*s.getLength(2)+row*s.getLength(0)+col;
                int oldIndex = (s.getOffset(0)+depth)*numElementsCols*numElementsRows +
                                (s.getOffset(1)+row)*numElementsCols +
                                (s.getOffset(2)+col);
                newBuff[newIndex] = buffer[oldIndex];
            }
        }
    }
    
    //put new memory block in sub-domain
    s.setBuffer(newBuff);
}//end copyBlock

void Decomposition::decompose3D(int* buffer, const int numElementsDepth,const int numElementsRows,const int numElementsCols){
#ifdef DEBUG
  fprintf(stdout, "entering decompose3D\n");
#endif
  int width = 4;  
  int blockDimDepth = static_cast<int>((numElementsDepth/(double)width)+.5);
  int blockDimHeight =  static_cast<int>((numElementsRows/(double)width)+.5);
  int blockDimWidth =  static_cast<int>((numElementsCols/(double)width)+.5);
#ifdef DEBUG
  fprintf(stdout, "clearing domain vector decompose3D\n");
#endif
  domain.clear();
  for(int i=0; i < width; ++i){
    for(int j=0; j < width; ++j){
      for(int k=0; k < width; ++k){
	SubDomain3D s;
        s.setLength(0, blockDimDepth);
        s.setOffset(0, blockDimDepth*i);
        s.setLength(1, blockDimHeight);
        s.setOffset(1, blockDimHeight*j);
        s.setLength(2, blockDimWidth);
        s.setOffset(2, blockDimWidth*k);
        
        //get data for this block from the buffer
        copyBlock(buffer,s,numElementsDepth, numElementsRows, numElementsCols);
      
	domain.push_back(s);
	s.setBuffer(NULL);
      }
    }
  }
 fprintf(stdout,"domain.size():%zu\n",domain.size()); 

}//end decompose3D


void Decomposition::decompose(int* buffer, const int numDimensions, const int numElements[3]){
fprintf(stderr,"decompose(%d dimensions, [%d][%d][%d]\n",numDimensions,numElements[0],numElements[1],numElements[2]);
  if(1 == numDimensions)
    decompose1D(buffer, numElements[0]);
  else if(2 == numDimensions)
    decompose2D(buffer, numElements[0],numElements[1]);
  else if(3 == numDimensions)
    decompose3D(buffer, numElements[0],numElements[1], numElements[2]);
}

