#include "Decomposition.h"
#include <stdio.h>
Decomposition::Decomposition(){ }

Decomposition::Decomposition(const int numSubDomains){ }

Decomposition::Decomposition(const Decomposition &d){ *this = d;  }

Decomposition::~Decomposition()
{
  for(size_t s = 0; s < domain.size(); ++s)
  {
    if(NULL != domain.at(s))
    {
      delete domain.at(s);
      domain.at(s)=NULL;
    }
  }
  domain.clear();
}

Decomposition& Decomposition::operator=(const Decomposition& d)
{
  this->domain.clear();
  for(size_t i=0; i < d.getNumSubDomains(); ++i)
  {
    SubDomain3D* s = new SubDomain3D(*(d.getSubDomain(i)));
    this->domain.push_back(s);
  }
  return *this;
}

const SubDomain3D* Decomposition::getSubDomain(const int index) const
{
  return this->domain.at(index);
}

SubDomain3D* Decomposition::popSubDomain()
{
  SubDomain3D* tmp = this->domain.back();
  this->domain.pop_back();
  return tmp;
}

SubDomain3D* Decomposition::getSubDomain(const int index)
{
  return this->domain.at(index);
}

size_t Decomposition::getNumSubDomains() const 
{
  return domain.size();
}

void Decomposition::addSubDomain(SubDomain3D* s)
{
  domain.push_back(s);
}

void Decomposition::decompose1D(int* buffer, 
                                const int numElementsX, 
                                const int stencil_size[3], 
                                const int iterations)
{
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

void Decomposition::decompose2D(int* buffer, 
                                const int numElementsRows, 
                                const int numElementsCols, 
                                const int stencil_size[3], 
                                const int iterations)
{
  if(0<numElementsRows&&0<numElementsCols){
    int numLeftX = numElementsRows;
    int numLeftY = numElementsCols;
    const int num_chunks = 8;
    int x_chunk_width = numElementsRows / num_chunks;
    int y_chunk_width = numElementsCols / num_chunks;

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


void Decomposition::copyBlock(  int* buffer, 
                                SubDomain3D* s, 
                                const int numElementsDepth, 
                                const int numElementsRows, 
                                const int numElementsCols)
{
  //subDomain should already have memory allocated.
  int *sBuff = s->getBuffer();

  if(NULL == sBuff)
  {
    fprintf(stderr,"copyBlock: subDomain has NULL Buffer.\n");
    return;
  }

  //stage sub-domain data into contiguous memory
  for(int depth = 0; depth < s->getLength(0); depth++)
  {
    for(int row = 0; row < s->getLength(1); row++)
    {
      for(int col = 0; col < s->getLength(2); col++)
      {
        //make sure index is valid
        int d = s->getOffset(0) + depth ;
        int r = s->getOffset(1) + row   ;
        int c = s->getOffset(2) + col   ;

        //don't copy if we aren't inside valid range
        if( d < 0 || d >= numElementsDepth  || 
            r < 0 || r >= numElementsRows   || 
            c < 0 || c >= numElementsCols   ) continue;

        int newIndex =  depth * s->getLength(1) * s->getLength(2) +
                        row * s->getLength(0) +
                        col ;

        int oldIndex =  d * numElementsCols * numElementsRows +
                        r * numElementsCols +
                        c ;

        sBuff[newIndex] = buffer[oldIndex];
      }
    }
  }
}//end copyBlock

void Decomposition::decompose3D(int* buffer, 
                                const int numElementsDepth,
                                const int numElementsRows,
                                const int numElementsCols, 
                                const int stencil_size[3], 
                                const int pyramidHeight)
{
  int width = 8;  
  int blockDimDepth   = static_cast<int>((numElementsDepth/(double)width)+.5);
  int blockDimHeight  = static_cast<int>((numElementsRows /(double)width)+.5);
  int blockDimWidth   = static_cast<int>((numElementsCols /(double)width)+.5);

  //calculate ghost zone
  int border[3] = { pyramidHeight*stencil_size[0],  
                    pyramidHeight*stencil_size[1], 
                    pyramidHeight*stencil_size[2]};

  domain.clear();
  for(int i=0; i < width; ++i)
  {
    for(int j=0; j < width; ++j)
    {
      for(int k=0; k < width; ++k)
      {
        int id[3]     = {i, j, k};

        //offset to account for ghost zone, may be negative
        int depthOff  = blockDimDepth  * i - border[0];
        int heightOff = blockDimHeight * j - border[1];
        int widthOff  = blockDimWidth  * k - border[2];

        //length may be too large when added to offset
        int depthLen  = blockDimDepth   + 2*border[0];
        int heightLen = blockDimHeight  + 2*border[1];
        int widthLen  = blockDimWidth   + 2*border[2];

        SubDomain3D* s = NULL;
        s= new SubDomain3D( id,         depthOff,   depthLen, 
                            heightOff,  heightLen,  widthOff, 
                            widthLen);

        //get data for this block from the buffer
        copyBlock(buffer, s, numElementsDepth, numElementsRows, numElementsCols);

        domain.push_back(s);
        s=NULL;
      }
    }
  }
}//end decompose3D


void Decomposition::decompose(  DTYPE* buffer, 
                                const int numDimensions, 
                                const int numElements[3], 
                                const int stencil_size[3], 
                                const int pyramidHeight)
{
  if(1 == numDimensions)
    decompose1D(buffer, numElements[0], stencil_size, pyramidHeight);
  else if(2 == numDimensions)
    decompose2D(buffer, numElements[0], numElements[1], stencil_size, pyramidHeight);
  else if(3 == numDimensions)
    decompose3D(buffer,       numElements[0], numElements[1], numElements[2], 
                stencil_size, pyramidHeight);
}

void printDecomposition(Decomposition& d)
{
  for(size_t s = 0; s < d.getNumSubDomains(); ++s)
  {
    fprintf(stderr, "s[%u] off[%d][%d][%d] len[%d][%d][%d].\n", 
            s,      
            d.getSubDomain(s)->getOffset(0), 
            d.getSubDomain(s)->getOffset(1), 
            d.getSubDomain(s)->getOffset(2), 
            d.getSubDomain(s)->getLength(0), 
            d.getSubDomain(s)->getLength(1), 
            d.getSubDomain(s)->getLength(2));
  }
}
