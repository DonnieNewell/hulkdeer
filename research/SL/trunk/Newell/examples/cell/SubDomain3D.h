#ifndef SUB_DOMAIN_3D_H
#define SUB_DOMAIN_3D_H

#include <vector>

/*
NEIGHBORS
(i, j, k) is current block
*/
enum NeighborTag {
  xNeighborBegin = 0,
  xFace0   =  xNeighborBegin,  // i-1, j  , k
  xFace1   =  1,  // i+1, j  , k
  xFace2   =  2,  // i  , j-1, k
  xFace3   =  3,  // i  , j+1, k
  xFace4   =  4,  // i  , j  , k-1
  xFace5   =  5,  // i  , j  , k+1
  xPole0   =  6,  // i-1, j-1, k
  xPole1   =  7,  // i-1, j+1, k
  xPole2   =  8,  // i+1, j+1, k
  xPole3   =  9,  // i+1, j-1, k
  xPole4   = 10,  // i-1, j  , k-1
  xPole5   = 11,  // i-1, j  , k+1
  xPole6   = 12,  // i+1, j  , k+1
  xPole7   = 13,  // i+1, j  , k-1
  xPole8   = 14,  // i  , j-1, k-1
  xPole9   = 15,  // i  , j-1, k+1
  xPole10  = 16,  // i  , j+1, k+1
  xPole11  = 17,  // i  , j+1, k-1
  xCorner0 = 18,  // i-1, j-1, k-1
  xCorner1 = 19,  // i-1, j-1, k+1
  xCorner2 = 20,  // i-1, j+1, k-1
  xCorner3 = 21,  // i-1, j+1, k+1
  xCorner4 = 22,  // i+1, j-1, k-1
  xCorner5 = 23,  // i+1, j-1, k+1
  xCorner6 = 24,  // i+1, j+1, k-1
  xCorner7 = 25,  // i+1, j+1, k+1
  xNeighborEnd = 26
};

NeighborTag &operator++(NeighborTag &n);
NeighborTag operator++(NeighborTag &n, int);

#define DTYPE int
class SubDomain3D{
  public:
    SubDomain3D();
    SubDomain3D(const SubDomain3D&);
    SubDomain3D(int[3],int, int, int, int, int, int, int, int, int, int[]);
    SubDomain3D& operator=(const SubDomain3D &);
    ~SubDomain3D();
    //needs to be set by compiler. DTYPE maybe?
    void setId(int,int,int, int, int, int);
    void setLength(int, int);
    void setOffset(int, int);
    void setNeighbors(std::vector<int>&);
    //needs to be set by compiler. DTYPE maybe?
    DTYPE* getBuffer()const;
    const int* getId()const;
    const int* getGridDim()const;
    int* getNeighbors();
    const int getLinIndex()const;
    int getLength(int)const;
    int getOffset(int)const;
    int getNeighborLoc(const NeighborTag)const;
    int getNeighborIndex(const NeighborTag);
  private:
    int getNeighborFace(const NeighborTag);
    int getNeighborPole(const NeighborTag);
    int getNeighborCorner(const NeighborTag);
    int threeDToLin(int,int,int,int,int,int) const;
    int gridDim[3];
    int id[3];
    int offset[3];
    int length[3];
    int neighbors[26];//ranks of all neighbors to exchange ghost zones with
    DTYPE* buffer;
};
const char* neighborString(NeighborTag);
void printNeighbors(const SubDomain3D *s);
#endif
