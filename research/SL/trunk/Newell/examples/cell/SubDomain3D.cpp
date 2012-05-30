#include "SubDomain3D.h"
#include <cstddef>
#include <cstdio>
#include <cstring>

SubDomain3D::SubDomain3D()
{
  id[0]     = -1  ;
  id[1]     = -1  ;
  id[2]     = -1  ;
  offset[0] = 0   ;
  offset[1] = 0   ;
  offset[2] = 0   ;
  length[0] = 0   ;
  length[1] = 0   ;
  length[2] = 0   ;
  buffer    = NULL;
}

SubDomain3D::SubDomain3D(const SubDomain3D& sd)
{
  (*this) = sd;
}

SubDomain3D::SubDomain3D( int id[3],    int xOffset,  int xLength, 
                          int yOffset,  int yLength,  
                          int zOffset,  int zLength)
{

  this->id[0]  = id[0];
  this->id[1]  = id[1];
  this->id[2]  = id[2];
  offset[0] = xOffset ;
  offset[1] = yOffset ;
  offset[2] = zOffset ;
  length[0] = xLength ;
  length[1] = yLength ;
  length[2] = zLength ;
  this->buffer = new DTYPE[xLength*yLength*zLength];
}

SubDomain3D::~SubDomain3D()
{
  if(this->buffer != NULL)  delete [] this->buffer;

  this->buffer=NULL;
}

void SubDomain3D::setId(int i, int j, int k)
{
  if(0<=i && 0<=j && 0<=k)
  {
    id[0] = i;
    id[1] = j;
    id[2] = k;
  }
}

void SubDomain3D::setOffset(int dim, int off)
{
    if(0 <= dim && 3 > dim && 0 <= off)
      offset[dim] = off;
}

void SubDomain3D::setLength(int dim, int len)
{
    if(0 <= dim && 3 > dim && 0 <= len)
      length[dim] = len;

}

DTYPE* SubDomain3D::getBuffer()const 
{
  return this->buffer;
}

const int* SubDomain3D::getId() const
{
  return id;
} 

int SubDomain3D::getOffset(int dim)const 
{

  if(0 <= dim && 3 > dim )
    return offset[dim];
  else
    return -1;
}

int SubDomain3D::getLength(int dim)const
{

  if(0 <= dim && 3 > dim )
    return length[dim];
  else
    return -1;
}

SubDomain3D& SubDomain3D::operator=(const SubDomain3D &sd) 
{

  // Only do assignment if RHS is a different object from this.
  if (this != &sd) {
    offset[0] = sd.getOffset(0);
    offset[1] = sd.getOffset(1);
    offset[2] = sd.getOffset(2);
    length[0] = sd.getLength(0);
    length[1] = sd.getLength(1);
    length[2] = sd.getLength(2);
    int size  = length[0]*length[1]*length[2];
    buffer    = new DTYPE[size];
    DTYPE*buf = sd.getBuffer();
    
    if(NULL!=buf)
    {
      memcpy( buffer, sd.getBuffer(), sizeof(DTYPE)*size);
    }
  }

  return *this;
}
