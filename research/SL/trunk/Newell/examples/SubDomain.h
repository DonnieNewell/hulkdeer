#ifndef SUB_DOMAIN_3D_H
#define SUB_DOMAIN_3D_H

#include <vector>
// TODO (den4gr) #include <boost/scoped_array.hpp>

/*
NEIGHBORS 3D
(i, j, k) is current block
 */
enum NeighborTag3D {
  x3DNeighborBegin = 0,
  x3DFace0 = x3DNeighborBegin, // i-1, j  , k
  x3DFace1 = 1, // i+1, j  , k
  x3DFace2 = 2, // i  , j-1, k
  x3DFace3 = 3, // i  , j+1, k
  x3DFace4 = 4, // i  , j  , k-1
  x3DFace5 = 5, // i  , j  , k+1
  x3DPole0 = 6, // i-1, j-1, k
  x3DPole1 = 7, // i-1, j+1, k
  x3DPole2 = 8, // i+1, j+1, k
  x3DPole3 = 9, // i+1, j-1, k
  x3DPole4 = 10, // i-1, j  , k-1
  x3DPole5 = 11, // i-1, j  , k+1
  x3DPole6 = 12, // i+1, j  , k+1
  x3DPole7 = 13, // i+1, j  , k-1
  x3DPole8 = 14, // i  , j-1, k-1
  x3DPole9 = 15, // i  , j-1, k+1
  x3DPole10 = 16, // i  , j+1, k+1
  x3DPole11 = 17, // i  , j+1, k-1
  x3DCorner0 = 18, // i-1, j-1, k-1
  x3DCorner1 = 19, // i-1, j-1, k+1
  x3DCorner2 = 20, // i-1, j+1, k-1
  x3DCorner3 = 21, // i-1, j+1, k+1
  x3DCorner4 = 22, // i+1, j-1, k-1
  x3DCorner5 = 23, // i+1, j-1, k+1
  x3DCorner6 = 24, // i+1, j+1, k-1
  x3DCorner7 = 25, // i+1, j+1, k+1
  x3DNeighborEnd = 26
};

/*
NEIGHBORS 2D
(i, j) is current block
 */
enum NeighborTag2D {
  x2DNeighborBegin = 0,
  x2DPole0 = x2DNeighborBegin, // i-1, j
  x2DPole1 = 1, // i, j+1
  x2DPole2 = 2, // i+1, j,
  x2DPole3 = 3, // i, j-1
  x2DCorner0 = 4, // i-1, j-1
  x2DCorner1 = 5, // i-1, j+1
  x2DCorner2 = 6, // i+1, j+1
  x2DCorner3 = 7, // i+1, j-1
  x2DNeighborEnd = 8
};

NeighborTag3D &operator++(NeighborTag3D &n);
NeighborTag3D operator++(NeighborTag3D &n, int);
NeighborTag2D &operator++(NeighborTag2D &n);
NeighborTag2D operator++(NeighborTag2D &n, int);

class SubDomain {
public:
  SubDomain();
  SubDomain(const SubDomain&);
  SubDomain(int* id, int zOffset, int zLength, int yOffset,
          int yLength, int xOffset, int xLength, int gridDepth, int gridHeight,
          int gridWidth, int* newNeighbors);
  SubDomain(int* id, int yOffset, int yLength, int xOffset,
          int xLength, int gridHeight, int gridWidth,
          int* newNeighbors);
  SubDomain& operator=(const SubDomain &);
  ~SubDomain();
  //needs to be set by compiler. DTYPE maybe?
  int getDimensionality() const;
  void setId(int i, int j, int k, int gridDepth, int gridHeight,
          int gridWidth);
  void setId(const unsigned int kI, const unsigned int kJ,
          const unsigned int kK);
  void setGridDim(const unsigned int kGridDepth,
          const unsigned int kGridHeight,
          const unsigned int kGridWidth);
  void setLength(int, int);
  void setBorder(const int kDim, const int kLen);
  void setOffset(int, int);
  void setNeighbors(std::vector<int>&);
  //needs to be set by compiler. DTYPE maybe?
  DTYPE* getBuffer()const;
  const int* getId()const;
  int* getId();
  const int* getGridDim()const;
  int* getNeighbors();
  int getLinIndex()const;
  int getLength(int)const;
  int getBorder(const int kDim)const;
  int getOffset(int)const;
  int getNeighborLoc(const int)const;
  int getNeighborIndex(const NeighborTag3D);
  int getNeighborIndex(const NeighborTag2D);
private:
  int getNeighborFace(const NeighborTag3D);
  int getNeighborPole(const NeighborTag3D);
  int getNeighborPole(const NeighborTag2D);
  int getNeighborCorner(const NeighborTag3D);
  int getNeighborCorner(const NeighborTag2D);
  int threeDToLin(int, int, int, int, int) const;
  int twoDToLin(int, int, int) const;
  int gridDim[3];
  int id[3];
  int offset[3];
  int length[3];
  int neighbors[26]; //ranks of all neighbors to exchange ghost zones with
  int border[3];
  // TODO (den4gr) use boost::scoped_array<DTYPE> buffer
  DTYPE* buffer;
};
const char* neighborString(NeighborTag3D);
const char* neighborString(NeighborTag2D);
void printNeighbors(const SubDomain *s);
void printSubDomain(const SubDomain *s);
#endif
