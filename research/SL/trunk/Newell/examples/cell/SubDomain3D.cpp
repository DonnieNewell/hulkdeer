#include "SubDomain3D.h"
#include <cstddef>
#include <cassert>
#include <cstdio>
#include <cstring>

SubDomain3D::SubDomain3D()
{
  gridDim[0]     = -1  ;
  gridDim[1]     = -1  ;
  gridDim[2]     = -1  ;
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
  memset(neighbors, 0, 26*sizeof(int));
}

SubDomain3D::SubDomain3D(const SubDomain3D& sd)
{
  (*this) = sd;
}

SubDomain3D::SubDomain3D( int id[3],      int xOffset,  int xLength,
    int yOffset,    int yLength,
    int zOffset,    int zLength, int gridDepth,
    int gridHeight, int gridWidth, int newNeighbors[26])
{

  this->gridDim[0]  = gridDepth ;
  this->gridDim[1]  = gridHeight;
  this->gridDim[2]  = gridWidth ;
  this->id[0]  = id[0];
  this->id[1]  = id[1];
  this->id[2]  = id[2];
  offset[0] = xOffset ;
  offset[1] = yOffset ;
  offset[2] = zOffset ;
  length[0] = xLength ;
  length[1] = yLength ;
  length[2] = zLength ;
  this->buffer = new DTYPE[xLength*yLength*zLength]();
  memcpy(static_cast<void*>(this->neighbors),
          static_cast<void*>(newNeighbors),
          26 * sizeof(int));
}

SubDomain3D::~SubDomain3D()
{
  if(this->buffer != NULL)  delete [] this->buffer;

  this->buffer=NULL;
}

void SubDomain3D::setId(int i, int j, int k,
    int gridDepth, int gridHeight, int gridWidth)
{
  if(0<=i && 0<=j && 0<=k && i<gridDepth && j<gridHeight && k<gridWidth)
  {
    gridDim[0] = gridDepth ;
    gridDim[1] = gridHeight;
    gridDim[2] = gridWidth ;
    id[0] = i;
    id[1] = j;
    id[2] = k;
  }
}

void SubDomain3D::setNeighbors( std::vector<int> &blockTable) {
  for (NeighborTag tag = xNeighborBegin; tag < xNeighborEnd; ++tag) {
    int linIndex = this->getNeighborIndex(tag);
    if (0 <= linIndex) {
      neighbors[tag] = blockTable.at(linIndex);
    } else {
      neighbors[tag] = linIndex;
    }
  }
}

void printNeighbors(const SubDomain3D *s) {
  const int* id = s->getId();
  int lin = s->getLinIndex();
  printf("neighbors of block[%d][%d][%d]\n linear index: %d\n",
          id[0], id[1], id[2], lin);
  for (NeighborTag tag = xFace0; tag <= xCorner7; ++tag) {
    int rank = s->getNeighborLoc(tag);
    printf("  n[%s]:%d\n", neighborString(tag), rank);
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

int SubDomain3D::threeDToLin(int i, int j, int k, int dim0, int dim1, int dim2) const {

  int linIndex =  i * dim2 * dim1  +
    j * dim2 +
    k ;
  return linIndex;
}

int SubDomain3D::getNeighborFace(const NeighborTag tag){
  const int i     = id[0];
  const int j     = id[1];
  const int k     = id[2];
  const int dimI  = gridDim[0];
  const int dimJ  = gridDim[1];
  const int dimK  = gridDim[2];
  int neighbor = -1;

  if (xFace0 == tag && 0 < i) {
    neighbor = threeDToLin(i-1, j, k, dimI, dimJ, dimK);

  } else if (xFace1 == tag && (dimI-1) > i) {
    neighbor = threeDToLin(i+1, j, k, dimI, dimJ, dimK);

  } else if (xFace2 == tag && 0 < j) {
    neighbor = threeDToLin(i, j-1, k, dimI, dimJ, dimK);

  } else if (xFace3 == tag && (dimJ-1) > j) {
    neighbor = threeDToLin(i, j+1, k, dimI, dimJ, dimK);

  } else if (xFace4 == tag && 0 < k) {
    neighbor = threeDToLin(i, j, k-1, dimI, dimJ, dimK);

  } else if (xFace5 == tag && (dimK-1) > k) {
    neighbor = threeDToLin(i, j, k+1, dimI, dimJ, dimK);
  }
  return neighbor;
}

int SubDomain3D::getNeighborPole(const NeighborTag tag){
  int neighbor = -1;
  const int i     = id[0];
  const int j     = id[1];
  const int k     = id[2];
  const int dimI  = gridDim[0];
  const int dimJ  = gridDim[1];
  const int dimK  = gridDim[2];

  if (xPole0 == tag && 0 < i && 0 < j) {
    neighbor = threeDToLin(i-1, j-1, k, dimI, dimJ, dimK);

  } else if (xPole1 == tag && 0 < i && (dimJ-1) > j) {
    neighbor = threeDToLin(i-1, j+1, k, dimI, dimJ, dimK);

  } else if (xPole2 == tag && (dimI-1) > i && (dimJ-1) > j) {
    neighbor = threeDToLin(i+1, j+1, k, dimI, dimJ, dimK);

  } else if (xPole3 == tag && (dimI-1) > i && 0 < j) {
    neighbor = threeDToLin(i+1, j-1, k, dimI, dimJ, dimK);

  } else if (xPole4 == tag && 0 < i && 0 < k) {
    neighbor = threeDToLin(i-1, j, k-1, dimI, dimJ, dimK);

  } else if (xPole5 == tag && 0 < i && (dimK-1) > k) {
    neighbor = threeDToLin(i-1, j, k+1, dimI, dimJ, dimK);

  } else if (xPole6 == tag && (dimI-1) > i && (dimK-1) > k) {
    neighbor = threeDToLin(i+1, j, k+1, dimI, dimJ, dimK);

  } else if (xPole7 == tag && (dimI-1) > i && 0 < k) {
    neighbor = threeDToLin(i+1, j, k-1, dimI, dimJ, dimK);

  } else if (xPole8 == tag && 0 < j && 0 < k) {
    neighbor = threeDToLin(i, j-1, k-1, dimI, dimJ, dimK);

  } else if (xPole9 == tag && 0 < j && (dimK-1) > k) {
    neighbor = threeDToLin(i, j-1, k+1, dimI, dimJ, dimK);

  } else if (xPole10 == tag && (dimJ-1) > j && (dimK-1) > k) {
    neighbor = threeDToLin(i, j+1, k+1, dimI, dimJ, dimK);

  } else if (xPole11 == tag && (dimJ-1) > j && 0 < k) {
    neighbor = threeDToLin(i, j+1, k-1, dimI, dimJ, dimK);
  }
  return neighbor;
}

int SubDomain3D::getNeighborCorner(const NeighborTag tag){
  const int i     = id[0];
  const int j     = id[1];
  const int k     = id[2];
  const int dimI  = gridDim[0];
  const int dimJ  = gridDim[1];
  const int dimK  = gridDim[2];
  int neighbor = -1;

  if (xCorner0 == tag && 0 < i && 0 < j && 0 < k) {
    neighbor = threeDToLin(i-1, j-1, k-1, dimI, dimJ, dimK);

  } else if (xCorner1 == tag && 0 < i && 0 < j && (dimK-1) > k) {
    neighbor = threeDToLin(i-1, j-1, k+1, dimI, dimJ, dimK);

  } else if (xCorner2 == tag && 0 < i && (dimJ-1) > j && 0 < k) {
    neighbor = threeDToLin(i-1, j+1, k-1, dimI, dimJ, dimK);

  } else if (xCorner3 == tag && 0 < i && (dimJ-1) > j && (dimK-1) > k) {
    neighbor = threeDToLin(i-1, j+1, k+1, dimI, dimJ, dimK);

  } else if (xCorner4 == tag && (dimI-1) > i && 0 < j && 0 < k) {
    neighbor = threeDToLin(i+1, j-1, k-1, dimI, dimJ, dimK);

  } else if (xCorner5 == tag && (dimI-1) > i && 0 < j && (dimK-1) > k) {
    neighbor = threeDToLin(i+1, j-1, k+1, dimI, dimJ, dimK);

  } else if (xCorner6 == tag && (dimI-1) > i && (dimJ-1) > j && 0 < k) {
    neighbor = threeDToLin(i+1, j+1, k-1, dimI, dimJ, dimK);

  } else if (xCorner7 == tag && (dimI-1) > i && (dimJ-1) > j && (dimK-1) > k) {
    neighbor = threeDToLin(i+1, j+1, k+1, dimI, dimJ, dimK);
  }
  return neighbor;
}

/* returns the linear index for neighbor specified by NeighborTag */
int SubDomain3D::getNeighborIndex(const NeighborTag tag){
   /* only a valid neighbor will be positive */
  int neighbor = -1;
  if (xPole0 > tag)
    neighbor = getNeighborFace(tag);
  else if (xFace5 < tag && xCorner0 > tag)
    neighbor = getNeighborPole(tag);
  else if (xPole11 < tag)
    neighbor = getNeighborCorner(tag);

  return neighbor;
}

const int SubDomain3D::getLinIndex() const {
  return threeDToLin(id[0], id[1], id[2], gridDim[0], gridDim[1], gridDim[2]);
}

int* SubDomain3D::getNeighbors() { return neighbors; }

const int* SubDomain3D::getGridDim() const { return gridDim; }

const int* SubDomain3D::getId() const { return id; }

/* returns rank of node where neighbor block is located */
int SubDomain3D::getNeighborLoc( const NeighborTag index)const {
    return neighbors[index];
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
    buffer    = new DTYPE[size]();
    DTYPE*buf = sd.getBuffer();

    if(NULL!=buf)
    {
      memcpy( buffer, sd.getBuffer(), sizeof(DTYPE)*size);
    }
  }

  return *this;
}

NeighborTag &operator++(NeighborTag &n) {
  assert(n != xNeighborEnd);
  n = static_cast<NeighborTag>(n + 1);
  return n;
}

NeighborTag operator++(NeighborTag &n, int) {
  assert(n != xNeighborEnd);
  ++n;
  return static_cast<NeighborTag>(n - 1);
}

const char* neighborString(NeighborTag neighbor) {
  switch (neighbor) {
  case xFace0:
    return "xFace0";
    break;
  case xFace1:
    return "xFace1";
    break;
  case xFace2:
    return "xFace2";
    break;
  case xFace3:
    return "xFace3";
    break;
  case xFace4:
    return "xFace4";
    break;
  case xFace5:
    return "xFace5";
    break;
  case xPole0:
    return "xPole0";
    break;
  case xPole1:
    return "xPole1";
    break;
  case xPole2:
    return "xPole2";
    break;
  case xPole3:
    return "xPole3";
    break;
  case xPole4:
    return "xPole4";
    break;
  case xPole5:
    return "xPole5";
    break;
  case xPole6:
    return "xPole6";
    break;
  case xPole7:
    return "xPole7";
    break;
  case xPole8:
    return "xPole8";
    break;
  case xPole9:
    return "xPole9";
    break;
  case xPole10:
    return "xPole10";
    break;
  case xPole11:
    return "xPole11";
    break;
  case xCorner0:
    return "xCorner0";
    break;
  case xCorner1:
    return "xCorner1";
    break;
  case xCorner2:
    return "xCorner2";
    break;
  case xCorner3:
    return "xCorner3";
    break;
  case xCorner4:
    return "xCorner4";
    break;
  case xCorner5:
    return "xCorner5";
    break;
  case xCorner6:
    return "xCorner6";
    break;
  case xCorner7:
    return "xCorner7";
    break;
  default:
    break;
  }
  return "invalid NeighborFlag";
}
