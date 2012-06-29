#ifndef COMM_H
#define COMM_H

#include "SubDomain.h"
#include "Cluster.h"
#include "mpi.h"
#ifndef WIN32
#include <sys/time.h>
#else
#include < time.h>
#endif

enum MPITagType {
  xDim        = 0, xLength  = 1 , xChildren     = 2,
  xDevice     = 3, xData    = 4 , xNumBlocks    = 5,
  xOffset     = 6, xWeight  = 7 , xWeightIndex  = 8,
  xEdgeWeight = 9, xId      = 10, xGridDim      = 11,
  xNeighborData   = 12, xNeighborIndex = 13, xNeighbors = 14 };

double secondsElapsed(struct timeval start, struct timeval stop);
void sendDataToNode(const int, int, SubDomain*);
void receiveNumberOfChildren(int, Cluster*);
void sendData(Node*);
void benchmarkNode(Node*, SubDomain*);
SubDomain* receiveDataFromNode(int, int*);
bool isSegmentFace(NeighborTag);
bool isSegmentPole(NeighborTag);
bool isSegmentCorner(NeighborTag);
void getCornerDimensions(NeighborTag, int*, int*, SubDomain*, const int[],
                          const bool);
void getFaceDimensions(NeighborTag, int*, int*, SubDomain*, const int[],
                        const bool);
void getPoleDimensions(NeighborTag, int*, int*, SubDomain*, const int[],
                        const bool);
void getSegmentDimensions(NeighborTag, int*, int*, SubDomain*,
                          const int [], const bool );
void copySegment(NeighborTag, SubDomain*, DTYPE*, const int [], const bool,
                  int*);
bool sendSegment(const NeighborTag, SubDomain*, DTYPE*, const int, MPI_Request*);
bool receiveSegment(const NeighborTag, SubDomain*, DTYPE*, const int, int*);
void sendNewGhostZones(const NeighborTag, Node*, const int [], MPI_Request*,
                        DTYPE***, int*, int* );
NeighborTag getOppositeNeighbor3D(const NeighborTag);
void receiveNewGhostZones(const NeighborTag, Node*, const int[], DTYPE***,
                          const int);
int getMaxSegmentSize(SubDomain*, const int[]);
void delete3DBuffer(const int, const int, const int, DTYPE***);
DTYPE*** new3DBuffer(const int, const int, const int);
void updateAllStaleData(Node*, const int[]);

#endif
