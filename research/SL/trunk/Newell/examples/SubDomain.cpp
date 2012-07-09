#include "SubDomain.h"
#include <cstddef>
#include <cassert>
#include <cstdio>
#include <cstring>
const int kNumNeighbors3D = 26;
const int kNumNeighbors2D = 8;

SubDomain::SubDomain() {
  gridDim[0] = -1;
  gridDim[1] = -1;
  gridDim[2] = -1;
  id[0] = -1;
  id[1] = -1;
  id[2] = -1;
  offset[0] = 0;
  offset[1] = 0;
  offset[2] = 0;
  length[0] = 0;
  length[1] = 0;
  length[2] = 0;
  buffer = NULL;
  memset(neighbors, 0, kNumNeighbors3D * sizeof (DTYPE));
}

SubDomain::SubDomain(const SubDomain& sd) {
  (*this) = sd;
}

SubDomain::SubDomain(int id[2], int yOffset, int yLength, int xOffset,
        int xLength, int gridHeight, int gridWidth,
        int newNeighbors[kNumNeighbors2D]) {
  this->gridDim[0] = gridHeight;
  this->gridDim[1] = gridWidth;
  this->gridDim[2] = -1;
  this->id[0] = id[0];
  this->id[1] = id[1];
  this->id[2] = -1;
  offset[0] = yOffset;
  offset[1] = xOffset;
  offset[2] = -1;
  length[0] = yLength;
  length[1] = xLength;
  length[2] = -1;
  this->buffer = new DTYPE[xLength * yLength]();
  memcpy(static_cast<void*> (this->neighbors), static_cast<void*> (newNeighbors),
          kNumNeighbors2D * sizeof (DTYPE));
}

SubDomain::SubDomain(int id[3], int zOffset, int zLength, int yOffset,
        int yLength, int xOffset, int xLength, int gridDepth, int gridHeight,
        int gridWidth, int newNeighbors[kNumNeighbors3D]) {
  this->gridDim[0] = gridDepth;
  this->gridDim[1] = gridHeight;
  this->gridDim[2] = gridWidth;
  this->id[0] = id[0];
  this->id[1] = id[1];
  this->id[2] = id[2];
  offset[0] = zOffset;
  offset[1] = yOffset;
  offset[2] = xOffset;
  length[0] = zLength;
  length[1] = yLength;
  length[2] = xLength;
  this->buffer = new DTYPE[xLength * yLength * zLength]();
  memcpy(static_cast<void*> (this->neighbors), static_cast<void*> (newNeighbors),
          kNumNeighbors3D * sizeof (DTYPE));
}

SubDomain::~SubDomain() {
  if (this->buffer != NULL) delete [] this->buffer;
  this->buffer = NULL;
}

void SubDomain::setId(int i, int j, int k, int gridDepth, int gridHeight,
        int gridWidth) {
  if (0 <= i && 0 <= j && 0 <= k &&
          i < gridDepth && j < gridHeight && k < gridWidth) {
    gridDim[0] = gridDepth;
    gridDim[1] = gridHeight;
    gridDim[2] = gridWidth;
    id[0] = i;
    id[1] = j;
    id[2] = k;
  }
}

int SubDomain::getDimensionality() const {
  const int k1D = 1;
  const int k2D = 2;
  const int k3D = 3;
  if (this->getId()[2] < 0) {
    if (this->getId()[1] < 0) {
      return k1D;
    } else {
      return k2D;
    }
  } else {
    return k3D;
  }
}

// TODO (donnie) differentiate based on 2d or 3d

void SubDomain::setNeighbors(std::vector<int> &blockTable) {
  const int kDimensionality = this->getDimensionality();
  const int k1D = 1;
  const int k2D = 2;
  const int k3D = 3;
  if (k1D == kDimensionality) {

  } else if (k2D == kDimensionality) {
    for (NeighborTag2D tag = x2DNeighborBegin; tag < x2DNeighborEnd; ++tag) {
      int linIndex = this->getNeighborIndex(tag);
      if (0 <= linIndex) {
        neighbors[tag] = blockTable.at(linIndex);
      } else {
        neighbors[tag] = linIndex;
      }
    }
  } else if (k3D == kDimensionality) {
    for (NeighborTag3D tag = x3DNeighborBegin; tag < x3DNeighborEnd; ++tag) {
      int linIndex = this->getNeighborIndex(tag);
      if (0 <= linIndex) {
        neighbors[tag] = blockTable.at(linIndex);
      } else {
        neighbors[tag] = linIndex;
      }
    }
  }
}

void printNeighbors(const SubDomain *s) {
  const int* id = s->getId();
  int lin = s->getLinIndex();
  printf("neighbors of block[%d][%d][%d]\n linear index: %d\n",
          id[0], id[1], id[2], lin);
  for (NeighborTag3D tag = x3DFace0; tag <= x3DCorner7; ++tag) {
    int rank = s->getNeighborLoc(tag);
    printf("  n[%s]:%d\n", neighborString(tag), rank);
  }
}

void SubDomain::setOffset(int dim, int off) {
  if (0 <= dim && 3 > dim && 0 <= off)
    offset[dim] = off;
}

void SubDomain::setLength(int dim, int len) {
  if (0 <= dim && 3 > dim && 0 <= len)
    length[dim] = len;
}

DTYPE* SubDomain::getBuffer() const {
  return this->buffer;
}

int SubDomain::threeDToLin(int i, int j, int k,
        int dim0, int dim1, int dim2) const {
  return i * dim2 * dim1 + j * dim2 + k;
}

int SubDomain::twoDToLin(int i, int j, int dim0, int dim1) const {
  return i * dim1 + j;
}

int SubDomain::getNeighborFace(const NeighborTag3D tag) {
  const int i = id[0];
  const int j = id[1];
  const int k = id[2];
  const int dimI = gridDim[0];
  const int dimJ = gridDim[1];
  const int dimK = gridDim[2];
  int neighbor = -1;

  if (x3DFace0 == tag && 0 < i) {
    neighbor = threeDToLin(i - 1, j, k, dimI, dimJ, dimK);

  } else if (x3DFace1 == tag && (dimI - 1) > i) {
    neighbor = threeDToLin(i + 1, j, k, dimI, dimJ, dimK);

  } else if (x3DFace2 == tag && 0 < j) {
    neighbor = threeDToLin(i, j - 1, k, dimI, dimJ, dimK);

  } else if (x3DFace3 == tag && (dimJ - 1) > j) {
    neighbor = threeDToLin(i, j + 1, k, dimI, dimJ, dimK);

  } else if (x3DFace4 == tag && 0 < k) {
    neighbor = threeDToLin(i, j, k - 1, dimI, dimJ, dimK);

  } else if (x3DFace5 == tag && (dimK - 1) > k) {
    neighbor = threeDToLin(i, j, k + 1, dimI, dimJ, dimK);
  }
  return neighbor;
}

int SubDomain::getNeighborPole(const NeighborTag3D tag) {
  int neighbor = -1;
  const int i = id[0];
  const int j = id[1];
  const int k = id[2];
  const int dimI = gridDim[0];
  const int dimJ = gridDim[1];
  const int dimK = gridDim[2];

  if (x3DPole0 == tag && 0 < i && 0 < j) {
    neighbor = threeDToLin(i - 1, j - 1, k, dimI, dimJ, dimK);

  } else if (x3DPole1 == tag && 0 < i && (dimJ - 1) > j) {
    neighbor = threeDToLin(i - 1, j + 1, k, dimI, dimJ, dimK);

  } else if (x3DPole2 == tag && (dimI - 1) > i && (dimJ - 1) > j) {
    neighbor = threeDToLin(i + 1, j + 1, k, dimI, dimJ, dimK);

  } else if (x3DPole3 == tag && (dimI - 1) > i && 0 < j) {
    neighbor = threeDToLin(i + 1, j - 1, k, dimI, dimJ, dimK);

  } else if (x3DPole4 == tag && 0 < i && 0 < k) {
    neighbor = threeDToLin(i - 1, j, k - 1, dimI, dimJ, dimK);

  } else if (x3DPole5 == tag && 0 < i && (dimK - 1) > k) {
    neighbor = threeDToLin(i - 1, j, k + 1, dimI, dimJ, dimK);

  } else if (x3DPole6 == tag && (dimI - 1) > i && (dimK - 1) > k) {
    neighbor = threeDToLin(i + 1, j, k + 1, dimI, dimJ, dimK);

  } else if (x3DPole7 == tag && (dimI - 1) > i && 0 < k) {
    neighbor = threeDToLin(i + 1, j, k - 1, dimI, dimJ, dimK);

  } else if (x3DPole8 == tag && 0 < j && 0 < k) {
    neighbor = threeDToLin(i, j - 1, k - 1, dimI, dimJ, dimK);

  } else if (x3DPole9 == tag && 0 < j && (dimK - 1) > k) {
    neighbor = threeDToLin(i, j - 1, k + 1, dimI, dimJ, dimK);

  } else if (x3DPole10 == tag && (dimJ - 1) > j && (dimK - 1) > k) {
    neighbor = threeDToLin(i, j + 1, k + 1, dimI, dimJ, dimK);

  } else if (x3DPole11 == tag && (dimJ - 1) > j && 0 < k) {
    neighbor = threeDToLin(i, j + 1, k - 1, dimI, dimJ, dimK);
  }
  return neighbor;
}

int SubDomain::getNeighborPole(const NeighborTag2D tag) {
  int neighbor = -1;
  const int i = id[0];
  const int j = id[1];
  const int dimI = gridDim[0];
  const int dimJ = gridDim[1];

  if (x2DPole0 == tag && 0 < i) {
    neighbor = twoDToLin(i - 1, j, dimI, dimJ);
  } else if (x2DPole1 == tag && (dimJ - 1) > j) {
    neighbor = twoDToLin(i, j + 1, dimI, dimJ);
  } else if (x2DPole2 == tag && (dimI - 1) > i) {
    neighbor = twoDToLin(i + 1, j, dimI, dimJ);
  } else if (x2DPole3 == tag && 0 < j) {
    neighbor = twoDToLin(i, j - 1, dimI, dimJ);
  }
  return neighbor;
}

int SubDomain::getNeighborCorner(const NeighborTag3D tag) {
  const int i = id[0];
  const int j = id[1];
  const int k = id[2];
  const int dimI = gridDim[0];
  const int dimJ = gridDim[1];
  const int dimK = gridDim[2];
  int neighbor = -1;

  if (x3DCorner0 == tag && 0 < i && 0 < j && 0 < k) {
    neighbor = threeDToLin(i - 1, j - 1, k - 1, dimI, dimJ, dimK);

  } else if (x3DCorner1 == tag && 0 < i && 0 < j && (dimK - 1) > k) {
    neighbor = threeDToLin(i - 1, j - 1, k + 1, dimI, dimJ, dimK);

  } else if (x3DCorner2 == tag && 0 < i && (dimJ - 1) > j && 0 < k) {
    neighbor = threeDToLin(i - 1, j + 1, k - 1, dimI, dimJ, dimK);

  } else if (x3DCorner3 == tag && 0 < i && (dimJ - 1) > j && (dimK - 1) > k) {
    neighbor = threeDToLin(i - 1, j + 1, k + 1, dimI, dimJ, dimK);

  } else if (x3DCorner4 == tag && (dimI - 1) > i && 0 < j && 0 < k) {
    neighbor = threeDToLin(i + 1, j - 1, k - 1, dimI, dimJ, dimK);

  } else if (x3DCorner5 == tag && (dimI - 1) > i && 0 < j && (dimK - 1) > k) {
    neighbor = threeDToLin(i + 1, j - 1, k + 1, dimI, dimJ, dimK);

  } else if (x3DCorner6 == tag && (dimI - 1) > i && (dimJ - 1) > j && 0 < k) {
    neighbor = threeDToLin(i + 1, j + 1, k - 1, dimI, dimJ, dimK);

  } else if (x3DCorner7 == tag && (dimI - 1) > i && (dimJ - 1) > j && (dimK - 1) > k) {
    neighbor = threeDToLin(i + 1, j + 1, k + 1, dimI, dimJ, dimK);
  }
  return neighbor;
}

int SubDomain::getNeighborCorner(const NeighborTag2D tag) {
  const int i = id[0];
  const int j = id[1];
  const int dimI = gridDim[0];
  const int dimJ = gridDim[1];
  int neighbor = -1;

  if (x2DCorner0 == tag && 0 < i && 0 < j) {
    neighbor = twoDToLin(i - 1, j - 1, dimI, dimJ);
  } else if (x2DCorner1 == tag && 0 < i && (dimJ - 1) > j) {
    neighbor = twoDToLin(i - 1, j + 1, dimI, dimJ);
  } else if (x2DCorner2 == tag && (dimI - 1) > i && (dimJ - 1) > j) {
    neighbor = twoDToLin(i + 1, j + 1, dimI, dimJ);
  } else if (x2DCorner3 == tag && (dimI - 1) > i && 0 < j) {
    neighbor = twoDToLin(i + 1, j - 1, dimI, dimJ);
  }
  return neighbor;
}

/* returns the linear index for neighbor specified by NeighborTag */
int SubDomain::getNeighborIndex(const NeighborTag3D tag) {
  /* only a valid neighbor will be positive */
  int neighbor = -1;
  if (x3DPole0 > tag)
    neighbor = getNeighborFace(tag);
  else if (x3DFace5 < tag && x3DCorner0 > tag)
    neighbor = getNeighborPole(tag);
  else if (x3DPole11 < tag)
    neighbor = getNeighborCorner(tag);

  return neighbor;
}

/* returns the linear index for neighbor specified by NeighborTag */
int SubDomain::getNeighborIndex(const NeighborTag2D tag) {
  /* only a valid neighbor will be positive */
  int neighbor = -1;
  if (x2DPole3 >= tag && x2DPole0 <= tag)
    neighbor = getNeighborPole(tag);
  else if (x2DCorner0 <= tag && x2DCorner3 >= tag)
    neighbor = getNeighborCorner(tag);
  return neighbor;
}

// TODO (donnie) need to differentiate 2d or 3d linear index

const int SubDomain::getLinIndex() const {
  const int kDimensionality = this->getDimensionality();
  const int k2D = 2;
  const int k3D = 3;
  if (k3D == kDimensionality) {
    int ret = threeDToLin(id[0], id[1], id[2], gridDim[0], gridDim[1], gridDim[2]);
    // printf("threeDToLin(id:%d, %d, %d gridDim:%d, %d, %d) == %d\n",id[0], id[1],
    //        id[2], gridDim[0], gridDim[1], gridDim[2], ret);
    return ret;
  } else if (k2D == kDimensionality) {
    return twoDToLin(id[0], id[1], gridDim[0], gridDim[1]);
  } else { // 1 dimension
    return id[0];
  }
}

int* SubDomain::getNeighbors() {
  return neighbors;
}

const int* SubDomain::getGridDim() const {
  return gridDim;
}

const int* SubDomain::getId() const {
  return id;
}

/* returns rank of node where neighbor block is located */
int SubDomain::getNeighborLoc(const int index)const {
  return neighbors[index];
}

int SubDomain::getOffset(int dim)const {

  if (0 <= dim && 3 > dim)
    return offset[dim];
  else
    return -1;
}

int SubDomain::getLength(int dim)const {

  if (0 <= dim && 3 > dim)
    return length[dim];
  else
    return -1;
}

SubDomain& SubDomain::operator=(const SubDomain &sd) {
  // Only do assignment if RHS is a different object from this.
  if (this != &sd) {
    offset[0] = sd.getOffset(0);
    offset[1] = sd.getOffset(1);
    offset[2] = sd.getOffset(2);
    length[0] = sd.getLength(0);
    length[1] = sd.getLength(1);
    length[2] = sd.getLength(2);
    int size = length[0] * length[1] * length[2];
    buffer = new DTYPE[size]();
    DTYPE*buf = sd.getBuffer();

    if (NULL != buf) {
      memcpy(buffer, sd.getBuffer(), sizeof (DTYPE) * size);
    }
  }
  return *this;
}

NeighborTag3D &operator++(NeighborTag3D &n) {
  assert(n != x3DNeighborEnd);
  n = static_cast<NeighborTag3D> (n + 1);
  return n;
}

NeighborTag3D operator++(NeighborTag3D &n, int) {
  assert(n != x3DNeighborEnd);
  ++n;
  return static_cast<NeighborTag3D> (n - 1);
}

NeighborTag2D &operator++(NeighborTag2D &n) {
  assert(n != x2DNeighborEnd);
  n = static_cast<NeighborTag2D> (n + 1);
  return n;
}

NeighborTag2D operator++(NeighborTag2D &n, int) {
  assert(n != x2DNeighborEnd);
  ++n;
  return static_cast<NeighborTag2D> (n - 1);
}

const char* neighborString(NeighborTag3D neighbor) {
  switch (neighbor) {
    case x3DFace0:
      return "x3DFace0";
      break;
    case x3DFace1:
      return "x3DFace1";
      break;
    case x3DFace2:
      return "x3DFace2";
      break;
    case x3DFace3:
      return "x3DFace3";
      break;
    case x3DFace4:
      return "x3DFace4";
      break;
    case x3DFace5:
      return "x3DFace5";
      break;
    case x3DPole0:
      return "x3DPole0";
      break;
    case x3DPole1:
      return "x3DPole1";
      break;
    case x3DPole2:
      return "x3DPole2";
      break;
    case x3DPole3:
      return "x3DPole3";
      break;
    case x3DPole4:
      return "x3DPole4";
      break;
    case x3DPole5:
      return "x3DPole5";
      break;
    case x3DPole6:
      return "x3DPole6";
      break;
    case x3DPole7:
      return "x3DPole7";
      break;
    case x3DPole8:
      return "x3DPole8";
      break;
    case x3DPole9:
      return "x3DPole9";
      break;
    case x3DPole10:
      return "x3DPole10";
      break;
    case x3DPole11:
      return "x3DPole11";
      break;
    case x3DCorner0:
      return "x3DCorner0";
      break;
    case x3DCorner1:
      return "x3DCorner1";
      break;
    case x3DCorner2:
      return "x3DCorner2";
      break;
    case x3DCorner3:
      return "x3DCorner3";
      break;
    case x3DCorner4:
      return "x3DCorner4";
      break;
    case x3DCorner5:
      return "x3DCorner5";
      break;
    case x3DCorner6:
      return "x3DCorner6";
      break;
    case x3DCorner7:
      return "x3DCorner7";
      break;
    default:
      break;
  }
  return "invalid 3DNeighborFlag";
}

const char* neighborString(NeighborTag2D neighbor) {
  switch (neighbor) {
    case x2DPole0:
      return "x2DPole0";
      break;
    case x2DPole1:
      return "x2DPole1";
      break;
    case x2DPole2:
      return "x2DPole2";
      break;
    case x2DPole3:
      return "x2DPole3";
      break;
    case x2DCorner0:
      return "x2DCorner0";
      break;
    case x2DCorner1:
      return "x2DCorner1";
      break;
    case x2DCorner2:
      return "x2DCorner2";
      break;
    case x2DCorner3:
      return "x2DCorner3";
    default:
      break;
  }
  return "invalid 2DNeighborFlag";
}

void printSubDomain(const SubDomain *s) {
  printf("SubDomain: linear_index:%d,", s->getLinIndex());
  printf(" dimensionality:%d,", s->getDimensionality());
  printf(" offset{%d, %d, %d}", s->getOffset(0), s->getOffset(1), s->getOffset(2));
  printf(" length{%d, %d, %d}\n", s->getLength(0), s->getLength(1), s->getLength(2));
  DTYPE* buffer = s->getBuffer();
  if (2 == s->getDimensionality()) {
    for (int i = 0; i < s->getLength(0); ++i) {
      printf("[%d]", i);
      for (int j = 0; j < s->getLength(1); ++j) {
        int index = i * s->getLength(1) + j;
        printf(" %.2d", buffer[index]);
      }
      printf("\n");
    }
  } else if (3 == s->getDimensionality()) {
    for (int i = 0; i < s->getLength(0); ++i) {
      printf("PLANE %d ***********************************",i);
      for (int j = 0; j < s->getLength(1); ++j) {
        printf("[%d]", i);
        for (int k = 0; k < s->getLength(2); ++k) {
          int index = i * s->getLength(1) * s->getLength(2) +
                        j * s->getLength(2) + k;
          printf(" %.2d", buffer[index]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}