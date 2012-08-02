#ifndef COMM_H
#define COMM_H

//#include "SubDomain.h"
#include "Cluster.h"
#include "mpi.h"
#ifndef WIN32
#include <sys/time.h>
#else
#include < time.h>
#endif

enum MPITagType {
  xDim = 0, xLength = 1, xChildren = 2,
  xDevice = 3, xData = 4, xNumBlocks = 5,
  xOffset = 6, xWeight = 7, xWeightIndex = 8,
  xEdgeWeight = 9, xId = 10, xGridDim = 11,
  xNeighborData = 12, xNeighborIndex = 13, xNeighbors = 14,
  xSubDomainEnvelope = 15
};

enum MPISubDomainBufferIndex {
  xID0 = 0, xID1 = 1, xID2 = 2, xGridDimension0 = 3, xGridDimension1 = 4,
  xGridDimension2 = 5, xDeviceIndex = 6, xNumberDimensions = 7, xLength0 = 8,
  xLength1 = 9, xLength2 = 10, xOffset0 = 11, xOffset1 = 12, xOffset2 = 13,
  xNeighborsStartIndex = 14
};

int getMPITagForSegment(NeighborTag2D);
int getMPITagForSegment(NeighborTag3D);
int getMPITagForSegmentData(NeighborTag2D);
int getMPITagForSegmentData(NeighborTag3D);
double secondsElapsed(struct timeval start, struct timeval stop);
void sendDataToNode(const int, int, SubDomain*);
void receiveNumberOfChildren(int, Cluster*);
void sendData(Node*);
void benchmarkNode(Node*, SubDomain*);
SubDomain* receiveDataFromNode(int, int*);
bool isSegmentFace(NeighborTag3D);
bool isSegmentPole(NeighborTag3D);
bool isSegmentCorner(NeighborTag3D);
bool isSegmentPole(NeighborTag2D);
bool isSegmentCorner(NeighborTag2D);
int getNumberDimensions(const SubDomain*);
void getCornerDimensions(NeighborTag3D, int*, int*, SubDomain*, const int[],
        const bool);
void getCornerDimensions(NeighborTag2D, int*, int*, SubDomain*, const int[],
        const bool);
void getFaceDimensions(NeighborTag3D, int*, int*, SubDomain*, const int[],
        const bool);
void getPoleDimensions(NeighborTag3D, int*, int*, SubDomain*, const int[],
        const bool);
void getPoleDimensions(NeighborTag2D, int*, int*, SubDomain*, const int[],
        const bool);
void getSegmentDimensions(NeighborTag3D, int*, int*, SubDomain*,
        const int [], const bool);
void getSegmentDimensions(NeighborTag2D, int*, int*, SubDomain*,
        const int [], const bool);
void copySegment(NeighborTag3D, SubDomain*, DTYPE*, const int [], const bool,
        int*);
void copySegment(NeighborTag2D, SubDomain*, DTYPE*, const int [], const bool,
        int*);
void copySegment(NeighborTag3D, Node*, SubDomain*, const int []);
void copySegment(NeighborTag2D, Node*, SubDomain*, const int []);
bool sendSegment(const NeighborTag3D, const int, SubDomain*, DTYPE*, const int, MPI_Request*);
bool sendSegment(const NeighborTag2D, const int, SubDomain*, DTYPE*, const int, MPI_Request*);
bool receiveSegment(const NeighborTag3D, const int, SubDomain*, DTYPE*, const int, int*);
bool receiveSegment(const NeighborTag2D, const int, SubDomain*, DTYPE*, const int, int*);
void sendNewGhostZones(const NeighborTag3D, Node*, const int [], MPI_Request*,
        int*, DTYPE***, int*, int*);
void sendNewGhostZones(const NeighborTag2D, Node*, const int [], MPI_Request*,
        int*, DTYPE***, int*, int*);
NeighborTag3D getOppositeNeighbor3D(const NeighborTag3D);
NeighborTag2D getOppositeNeighbor2D(const NeighborTag2D);
void exchangeGhostZones2D(Node*, const int[], DTYPE***, MPI_Request*);
void exchangeGhostZones3D(Node*, const int[], DTYPE***, MPI_Request*);
void receiveNewGhostZones(const NeighborTag3D, Node*, const int[], DTYPE***,
        const int);
void receiveNewGhostZones(const NeighborTag2D, Node*, const int[], DTYPE***,
        const int);
int getMaxSegmentSize(const SubDomain*, const int[], const int);
int getMaxSegmentSize2D(const SubDomain*, const int[]);
int getMaxSegmentSize3D(const SubDomain*, const int[]);
void delete3DBuffer(const int, const int, DTYPE***);
DTYPE*** new3DBuffer(const int, const int, const int);
void updateAllStaleData(Node*, const int[]);
void cleanupComm(const int);

#endif
