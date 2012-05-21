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
  for(size_t s = 0; s<domain.size();++s)
  {
    if(NULL != domain.at(s))
    {
      delete domain.at(s);
      domain.at(s)=NULL;
    }
  }
  domain.clear();
}
Decomposition& Decomposition::operator=(const Decomposition& d){
  this->domain.clear();
  for ( size_t i=0; i<d.getNumSubDomains(); ++i){
    SubDomain3D* s = new SubDomain3D(*(d.getSubDomain(i)));
    this->domain.push_back(s);
  }
  return *this;
}
const SubDomain3D* Decomposition::getSubDomain(const int index) const{
  return this->domain.at(index);
}
SubDomain3D* Decomposition::popSubDomain(){
  SubDomain3D* tmp = this->domain.back();
  this->domain.pop_back();
  return tmp;
}
SubDomain3D* Decomposition::getSubDomain(const int index){
  return this->domain.at(index);
}

size_t Decomposition::getNumSubDomains() const {
  return domain.size();
}
void Decomposition::addSubDomain(SubDomain3D* s){

  domain.push_back(s);
}
void Decomposition::decompose1D(int* buffer, const int numElementsX, const int stencil_size[3], const int iterations){
  if(0<numElementsX){
    int numLeftX=numElementsX;
    const int num_chunks = 64;
    int stride = numElementsX/num_chunks;

    for(size_t i=0; i < domain.size(); ++i){
      SubDomain3D* s = domain.at(i);


      s->setLength(0, stride);
      s->setOffset(0, numElementsX - numLeftX);

      numLeftX -= s->getLength(0);
    } //end for
  }//end if
}//end decompose1D

void Decomposition::decompose2D(int* buffer, const int numElementsRows,const int numElementsCols, const int stencil_size[3], const int iterations){
  if(0<numElementsRows&&0<numElementsCols){
    int numLeftX=numElementsRows;
    int numLeftY=numElementsCols;
    const int num_chunks = 8;
    int x_chunk_width = numElementsRows/num_chunks;
    int y_chunk_width = numElementsCols/num_chunks;

    for(size_t i=0; i < domain.size(); ++i){
      SubDomain3D* s = domain.at(i);
      s->setLength(0, x_chunk_width);
      s->setOffset(0, numElementsRows - numLeftX);
      s->setLength(1, y_chunk_width);
      s->setOffset(1, numElementsCols - numLeftY);

      numLeftX -= s->getLength(0);
      numLeftY -= s->getLength(1);
    } 
  }
}//end decompose2D


void Decomposition::copyBlock( int* buffer, SubDomain3D* s, const int numElementsDepth, const int numElementsRows, const int numElementsCols){

  //subDomain should already have memory allocated.
  int *sBuff = s->getBuffer();

  if(NULL==sBuff)
  {
    fprintf(stderr,"copyBlock: subDomain has NULL Buffer.\n");
    return;
  }
  //stage sub-domain data into contiguous memory
  for(int depth = 0; depth < s->getLength(0); depth++){
    for(int row = 0; row < s->getLength(1); row++){
      for(int col = 0; col < s->getLength(2); col++){
        int newIndex = depth*s->getLength(1)*s->getLength(2)+row*s->getLength(0)+col;
        int oldIndex = (s->getOffset(0)+depth)*numElementsCols*numElementsRows +
          (s->getOffset(1)+row)*numElementsCols +
          (s->getOffset(2)+col);
        sBuff[newIndex] = buffer[oldIndex];
      }
    }
  }

}//end copyBlock

void Decomposition::decompose3D(int* buffer, const int numElementsDepth,const int numElementsRows,const int numElementsCols, const int stencil_size[3], const int iterations){
#ifdef DEBUG
  fprintf(stdout, "entering decompose3D\n");
#endif
  int width = 8;  
  int blockDimDepth = static_cast<int>((numElementsDepth/(double)width)+.5);
  int blockDimHeight =  static_cast<int>((numElementsRows/(double)width)+.5);
  int blockDimWidth =  static_cast<int>((numElementsCols/(double)width)+.5);

  //calculate ghost zone
  int border[3] = {iterations*stencil_size[0],iterations*stencil_size[1],iterations*stencil_size[2]};

#ifdef DEBUG
  fprintf(stdout, "clearing domain vector decompose3D\n");
#endif
  domain.clear();
  for(int i=0; i < width; ++i){
    for(int j=0; j < width; ++j){
      for(int k=0; k < width; ++k){
        int depthOff = blockDimDepth*i;
        //int depthOff = blockDimDepth*i-border[0];
        depthOff = max(depthOff,0);
        int heightOff = blockDimHeight*j;
        //int heightOff = blockDimHeight*j-border[1];
        heightOff = max(heightOff,0);
        int widthOff = blockDimWidth*k;
        //int widthOff = blockDimWidth*k-border[2];
        widthOff = max(widthOff,0);
        int depthLen= blockDimDepth;
        //int depthLen= blockDimDepth + 2*border[0];
        depthLen = (depthOff+depthLen > numElementsDepth)?numElementsDepth-depthOff:depthLen;
        int heightLen = blockDimHeight;
        //int heightLen = blockDimHeight + 2*border[1];
        heightLen = (heightOff+heightLen > numElementsDepth)?numElementsDepth-heightOff:heightLen;
        int widthLen = blockDimWidth;
        //int widthLen = blockDimWidth + 2*border[2];
        widthLen = (widthOff+widthLen > numElementsDepth)?numElementsDepth-widthOff:widthLen;
        SubDomain3D* s = NULL;
        s= new SubDomain3D(depthOff, depthLen, heightOff, heightLen, widthOff, widthLen);

        //get data for this block from the buffer
        copyBlock(buffer,s,numElementsDepth, numElementsRows, numElementsCols);

        domain.push_back(s);
        s=NULL;
      }
    }
  }
#ifdef DEBUG
  fprintf(stdout,"domain.size():%zu\n",domain.size()); 
#endif

}//end decompose3D


void Decomposition::decompose(DTYPE* buffer, const int numDimensions, const int numElements[3], const int stencil_size[3], const int iterations){
#ifdef DEBUG
  fprintf(stderr,"decompose(%d dimensions, [%d][%d][%d]\n",numDimensions,numElements[0],numElements[1],numElements[2]);
#endif
  if(1 == numDimensions)
    decompose1D(buffer, numElements[0], stencil_size, iterations);
  else if(2 == numDimensions)
    decompose2D(buffer, numElements[0],numElements[1], stencil_size, iterations);
  else if(3 == numDimensions)
    decompose3D(buffer, numElements[0],numElements[1], numElements[2], stencil_size, iterations);
}
void printDecomposition(Decomposition& d)
{
  for(size_t s = 0; s < d.getNumSubDomains(); ++s)
  {
    fprintf(stderr, "s[%u] off[%d][%d][%d] len[%d][%d][%d].\n",s,d.getSubDomain(s)->getOffset(0),d.getSubDomain(s)->getOffset(1),d.getSubDomain(s)->getOffset(2),d.getSubDomain(s)->getLength(0),d.getSubDomain(s)->getLength(1),d.getSubDomain(s)->getLength(2));


  }
}
