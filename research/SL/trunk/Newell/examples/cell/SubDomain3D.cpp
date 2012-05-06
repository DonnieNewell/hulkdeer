#include "SubDomain3D.h"
#include <cstddef>
#include <cstdio>
#include <cstring>

SubDomain3D::SubDomain3D(){
  offset[0]=0;
  offset[1]=0;
  offset[2]=0;
  length[0]=0;
  length[1]=0;
  length[2]=0;
  buffer = NULL;
}

SubDomain3D::SubDomain3D(const SubDomain3D& sd){
  (*this) = sd;
}
SubDomain3D::SubDomain3D(int xOffset,int xLength,int yOffset,int yLength,int zOffset,int zLength){


  offset[0]=xOffset;
  offset[1]=yOffset;
  offset[2]=zOffset;
  length[0]=xLength;
  length[1]=yLength;
  length[2]=zLength;
  //DTYPE needed
  this->buffer = new int[xLength*yLength*zLength];
}
  SubDomain3D::~SubDomain3D(){
    if(this->buffer != NULL)
    {
#ifdef DEBUG
      fprintf(stderr, "deleting 0x%p.\n",this->buffer);
#endif
      delete [] this->buffer;
    }
    this->buffer=NULL;
  }

  void SubDomain3D::setOffset(int dim, int off){
    if(0 <= dim && 3 > dim && 0 <= off)
      offset[dim] = off;
  }
  void SubDomain3D::setLength(int dim, int len){
    if(0 <= dim && 3 > dim && 0 <= len)
      length[dim] = len;

  }
int* SubDomain3D::getBuffer()const {
  return this->buffer;
}
int SubDomain3D::getOffset(int dim)const {

  if(0 <= dim && 3 > dim )
    return offset[dim];
  else
    return -1;
}
int SubDomain3D::getLength(int dim)const{

  if(0 <= dim && 3 > dim )
    return length[dim];
  else
    return -1;
}
SubDomain3D& SubDomain3D::operator=(const SubDomain3D &sd) {

  // Only do assignment if RHS is a different object from this.
  if (this != &sd) {
    offset[0]=sd.getOffset(0);
    offset[1]=sd.getOffset(1);
    offset[2]=sd.getOffset(2);
    length[0]=sd.getLength(0);
    length[1]=sd.getLength(1);
    length[2]=sd.getLength(2);
    int size=length[0]*length[1]*length[2];
    buffer = new int[size];
    int*buf = sd.getBuffer();
    if(NULL!=buf)
    {
#ifdef DEBUG
      fprintf(stderr, "copying %lu bytes subdomain.\n",sizeof(int)*size);
#endif
      memcpy(buffer,sd.getBuffer(),sizeof(int)*size);
    }
  }

  return *this;
}
