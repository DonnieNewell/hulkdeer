/*
Copyright 2012 Donald Newell
*/
#include <stdlib.h>
#include <stdio.h>
#ifndef WIN32
#include <sys/time.h>
#else
#include < time.h>
#endif
#include <limits>
#include "mpi.h"
#include "cell.h"
#include "ompCell.h"
#include "Cluster.h"
#include "Decomposition.h"
#include "Balancer.h"
#include "../Model.h"
#define DTYPE int
#define PYRAMID_HEIGHT 1

enum MPITagType {
  xDim        = 0, xLength  = 1 , xChildren     = 2,
  xDevice     = 3, xData    = 4 , xNumBlocks    = 5,
  xOffset     = 6, xWeight  = 7 , xWeightIndex  = 8,
  xEdgeWeight = 9, xId      = 10, xGridDim      = 11,
  xNeighbor   = 12 };

const int kCPUIndex = -1;

double secondsElapsed(struct timeval start, struct timeval stop) {
    return static_cast<double>((stop.tv_sec - start.tv_sec) +
                                (stop.tv_usec - start.tv_usec)/1000000.0);
}

void sendDataToNode(int rank, int device, SubDomain3D* s) {
  // first send number of dimensions
  int numDim  = 0;
  MPI_Request reqs[7];
  int length[3];
  int offset[3];
  const int* kTmpId      = s->getId();
  const int* kTmpGridDim = s->getGridDim();
  int tmpId[3] = {kTmpId[0], kTmpId[1], kTmpId[2]};
  int tmpGridDim[3] = {kTmpGridDim[0], kTmpGridDim[1], kTmpGridDim[2]};
  MPI_Isend(static_cast<void*>(tmpId), 3, MPI_INT, rank, xId,
            MPI_COMM_WORLD, &reqs[5]);
  MPI_Isend(static_cast<void*>(tmpGridDim), 3, MPI_INT, rank, xGridDim,
            MPI_COMM_WORLD, &reqs[6]);
  MPI_Isend(static_cast<void*>(&device), 1, MPI_INT, rank, xDevice,
            MPI_COMM_WORLD, &reqs[0]);
  for (int i = 0; i < 3; ++i) {
    length[i] = s->getLength(i);
    offset[i] = s->getOffset(i);
    if (length[i] > 0) numDim++;
  }
  MPI_Isend(static_cast<void*>(&numDim), 1, MPI_INT, rank, xDim,
            MPI_COMM_WORLD, &reqs[1]);
  MPI_Isend(static_cast<void*>(length), 3, MPI_INT, rank, xLength,
            MPI_COMM_WORLD, &reqs[2]);
  MPI_Isend(static_cast<void*>(offset), 3, MPI_INT, rank, xOffset,
            MPI_COMM_WORLD, &reqs[3]);

  // third send data
  // first we have to stage the data into contiguous memory
  int total_size = 1;
  for (int i = 0; i < numDim; ++i) {
    total_size *= length[i];
  }
  MPI_Isend(static_cast <void*>(s->getBuffer()), total_size, MPI_INT, rank,
             xData, MPI_COMM_WORLD, &reqs[4]);
  MPI_Waitall(7, reqs, MPI_STATUSES_IGNORE);
}

void getNumberOfChildren(int* numChildren) {
  /* check to see how many NVIDIA GPU'S ARE AVAILABLE */
  cudaError_t err = cudaGetDeviceCount(numChildren);
  if (cudaSuccess == cudaErrorNoDevice) {
    *numChildren = 0;
  } else if (cudaSuccess != err) {
    fprintf(stderr, "error detecting cuda-enabled devices\n");
    *numChildren = 0;
  }
}

void sendNumberOfChildren(const int dest_rank, const int numChildren) {
  MPI_Request req;
  int sendNumChildrenBuffer = numChildren;
  MPI_Isend(static_cast<void*>(&sendNumChildrenBuffer), 1, MPI_INT, dest_rank,
            xChildren, MPI_COMM_WORLD, &req);
  MPI_Waitall(1, &req, MPI_STATUSES_IGNORE);
}

void receiveNumberOfChildren(int numTasks, Cluster* cluster) {
  MPI_Request *reqs = new MPI_Request[numTasks - 1];

  int* numChildren = new int[numTasks-1];

  for (int i = 0; i < numTasks-1; i++) {
    // receive next count
    MPI_Irecv(static_cast<void*>(numChildren + i), 1, MPI_INT, i+1, xChildren,
              MPI_COMM_WORLD, &(reqs[i]));
  }
  MPI_Waitall(numTasks-1, reqs, MPI_STATUSES_IGNORE);
  for (int task = 0; task < numTasks-1; task++) {
    cluster->getNode(task+1).setNumChildren(numChildren[task]);
  }

  delete reqs;
  reqs = NULL;
  delete numChildren;
  numChildren = NULL;
}

void sendData(Node* n) {
  // count how many task blocks, total, are going to be sent
  int total = n->numSubDomains();
  for (unsigned int child = 0; child < n->getNumChildren(); ++child) {
    total += n->getChild(child).numSubDomains();
  }

  // send node number of blocks
  MPI_Request req;
  MPI_Isend(static_cast<void*>(&total), 1, MPI_INT, n->getRank(), xNumBlocks,
            MPI_COMM_WORLD, &req);

  int device = -1;
  for (unsigned int i = 0; i < n->numSubDomains(); ++i) {
    sendDataToNode(n->getRank(), device, n->getSubDomain(i));
  }
  for (unsigned int child = 0; child < n->getNumChildren(); ++child) {
    for (unsigned int i = 0; i < n->getChild(child).numSubDomains(); ++i) {
      sendDataToNode(n->getRank(), child, n->getChild(child).getSubDomain(i));
    }
  }
  // wait for first send to finish
  MPI_Waitall(1, &req, MPI_STATUSES_IGNORE);
}

void benchmarkNode(Node* n, SubDomain3D* s) {
  struct timeval start, end;
  double total_sec = 0.0;
  gettimeofday(&start, NULL);
  // send task block to every device on that node
  sendDataToNode(n->getRank(), -1, s);
  gettimeofday(&end, NULL);

  total_sec = secondsElapsed(start, end);
  // how fast is the connection between root and child nodes
  // multiply by 2 to account for there and back
  n->setEdgeWeight(1/(2*total_sec));
  // receive results for each device
  unsigned int total = n->getNumChildren()+1;

  MPI_Request req[2];
  double *task_per_sec = new double[total];
  double *edge_weight  = new double[total-1];

  MPI_Irecv(static_cast<void*>(task_per_sec), total, MPI_DOUBLE, n->getRank(),
            xWeight, MPI_COMM_WORLD, &req[0]);
  MPI_Irecv(static_cast<void*>(edge_weight), total-1, MPI_DOUBLE, n->getRank(),
            xEdgeWeight, MPI_COMM_WORLD, &req[1]);
  MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

  // set the appropriate fields in the node and its children
  for (unsigned int device = 0; device < total; ++device) {
    double weight = task_per_sec[device];
    if (device == 0) {
      // the first weight is for the cpu
      fprintf(stderr, "setting node[%d] weight to %f.\n", n->getRank(), weight);
      n->setWeight(weight);
    } else {
      double edgeWeight = edge_weight[device-1];
      fprintf(stderr, "setting node[%d].child[%d] weight to %f.\n",
              n->getRank(), device-1, weight);
      n->getChild(device-1).setWeight(weight);
      n->getChild(device-1).setEdgeWeight(edgeWeight);
    }
  }
  // clean up
  delete [] task_per_sec;
  task_per_sec = NULL;
  delete [] edge_weight;
  edge_weight = NULL;
}

/* output variables: buf, size */
SubDomain3D* receiveDataFromNode(int rank, int* device) {
  MPI_Request reqs[6];
  int numDim =  0;
  int id[3]     = {-1, -1, -1};
  int gridDim[3]= {-1, -1, -1};
  int length[3];
  int offset[3];
  // receive dimensionality of data
  MPI_Irecv(static_cast<void*>(id), 3, MPI_INT, rank, xId, MPI_COMM_WORLD,
            &reqs[4]);
  MPI_Irecv(static_cast<void*>(gridDim), 3, MPI_INT, rank, xGridDim,
            MPI_COMM_WORLD,  &reqs[5]);
  MPI_Irecv(static_cast<void*>(device), 1, MPI_INT, rank, xDevice,
            MPI_COMM_WORLD,  &reqs[0]);
  MPI_Irecv(static_cast<void*>(&numDim), 1, MPI_INT, rank, xDim,
            MPI_COMM_WORLD,  &reqs[1]);

  // receive size of data
  MPI_Irecv(static_cast<void*>(length), 3, MPI_INT, rank, xLength,
            MPI_COMM_WORLD,  &reqs[2]);
  MPI_Irecv(static_cast<void*>(offset), 3, MPI_INT, rank, xOffset,
            MPI_COMM_WORLD,  &reqs[3]);

  MPI_Waitall(6, reqs, MPI_STATUSES_IGNORE);

  SubDomain3D *s = new SubDomain3D(id, offset[0], length[0], offset[1],
                                   length[1], offset[2], length[2], gridDim[0],
                                   gridDim[1], gridDim[2]);
  int size = 1;
  for (int i =0; i < numDim; ++i) {
    s->setLength(i, length[i]);
    s->setOffset(i, offset[i]);
    size *= length[i];
  }

  // allocates data memory and sets up 2d and 3d data pointers
  // initData(length);

  // needs to be set by compiler. DTYPE maybe?
  // if the buffer is already allocated, that the size is correct.
  // if (s.getBuffer()==NULL)
  //    s.setBuffer(new int[size]);

  // MPI_INT needs to be set by compiler. DTYPE maybe?
  MPI_Irecv(static_cast<void*>(s->getBuffer()), size, MPI_INT, rank, xData,
            MPI_COMM_WORLD,  &reqs[0]);

  // wait for everything to finish
  MPI_Waitall(1, reqs, MPI_STATUSES_IGNORE);
  return s;
}

void processSubDomain(int device, SubDomain3D *task, int timesteps,
                      int bornMin, int bornMax, int dieMin, int dieMax) {
  // DTYPE?
  DTYPE* buff = task->getBuffer();
  int depth = task->getLength(0);
  int height = task->getLength(1);
  int width = task->getLength(2);
  struct timeval start, end;
  if (-1 == device) {
    // run on CPU
    runOMPCell(buff, depth, height, width, timesteps, bornMin, bornMax, dieMin,
                dieMax, device);
  } else {
    // run on GPU
    gettimeofday(&start, NULL);
    runCell(buff, depth, height, width, timesteps, bornMin, bornMax, dieMin,
            dieMax, device);
    gettimeofday(&end, NULL);
    double kerneltime = secondsElapsed(start, end);
  }
}

void receiveData(int rank, Node* n, bool processNow, int pyramidHeight = 1,
                  int bornMin = 0, int bornMax = 0, int dieMin = 0,
                  int dieMax = 0) {
  // receive number of task blocks that will be sent
  int numTaskBlocks = 0;
  MPI_Status stat;
  MPI_Recv(static_cast<void*>(&numTaskBlocks), 1, MPI_INT, rank, xNumBlocks,
            MPI_COMM_WORLD, &stat);
  struct timeval start, end;
  double receiveDataTime = 0.0, processBlockTime = 0.0;
  for (int block = 0; block < numTaskBlocks; ++block) {
    SubDomain3D* s = NULL;
    int device = -1;
    gettimeofday(&start, NULL);
    s = receiveDataFromNode(rank, &device);
    gettimeofday(&end, NULL);
    receiveDataTime += secondsElapsed(start, end);
    if (-1 == device) {
      if (processNow) {
        gettimeofday(&start, NULL);
        processSubDomain(device, s, pyramidHeight, bornMin, bornMax,
                          dieMin, dieMax);
        gettimeofday(&end, NULL);
        processBlockTime+= secondsElapsed(start, end);
      }
      // add block to cpu queue
      n->addSubDomain(s);
    } else {
      if (processNow) {
        gettimeofday(&start, NULL);
        processSubDomain(device, s, pyramidHeight, bornMin, bornMax,
                          dieMin, dieMax);
        gettimeofday(&end, NULL);
        processBlockTime += secondsElapsed(start, end);
      }
      // add block to gpu queue
      n->getChild(device).addSubDomain(s);
    }
  }
  fprintf(stderr, "[%d] comm. time %f, process time %f.\n",
          n->getRank(), receiveDataTime, processBlockTime);
  runCellCleanup();
}

double benchmarkPCIBus(SubDomain3D* pS, int gpuIndex) {
  struct timeval start, end;
  double total = 0.0;
  gettimeofday(&start, NULL);
  DTYPE* devBuffer  = NULL;
  int    currDevice = -1;
  cudaGetDevice(&currDevice);
  if (currDevice != gpuIndex) {
    if (cudaSetDevice(gpuIndex)!= cudaSuccess) {
      fprintf(stderr, "ERROR: couldn't set device to %d\n", gpuIndex);
      return -1.0;
    }
  }
  size_t size = sizeof(DTYPE) * pS->getLength(0) *
                pS->getLength(1) * pS->getLength(2);
  cudaMalloc(&devBuffer, size);
  cudaMemcpy(static_cast<void*>(devBuffer), static_cast<void*>(pS->getBuffer()),
              size, cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<void*>(pS->getBuffer()), static_cast<void*>(devBuffer),
              size, cudaMemcpyDeviceToHost);
  cudaFree(devBuffer);
  devBuffer = NULL;
  gettimeofday(&end, NULL);
  total = secondsElapsed(start, end);
  return 1 / total;
}

void benchmarkMyself(Node* n, SubDomain3D* pS, int timesteps, int bornMin,
                      int bornMax, int dieMin, int dieMax) {
  // receive results for each device
  unsigned int total = n->getNumChildren()+1;
  MPI_Request req[2];
  double *weight = new double[total];
  double *edgeWeight = new double[total-1];
  SubDomain3D *s = NULL;
  int rank = -2;
  if (pS == NULL) {
    s = receiveDataFromNode(0, &rank);
    if (-1 != rank) {
      fprintf(stderr, "data should be sent to device: -1, not:%d\n", rank);
    }
  } else {
    s = pS;
  }
  for (unsigned int device = 0; device < total; ++device) {
    int iterations = 100;
    struct timeval start, end;
    double total_sec = 0.0;
    gettimeofday(&start, NULL);
    for (int itr = 0; itr < iterations; ++itr) {
      processSubDomain(device-1, s, timesteps, bornMin, bornMax, dieMin,
                        dieMax);
    }
    gettimeofday(&end, NULL);
    total_sec = secondsElapsed(start, end);
    weight[device] = iterations/total_sec;
    fprintf(stderr, "[%d]device:%d of %d processes %f iter/sec.\n",
            n->getRank(), device-1, total, weight[device]);
    if (device == 0) {
      n->setWeight(weight[device]);
      n->setEdgeWeight(numeric_limits<double>::max());
    } else {
      n->getChild(device-1).setWeight(weight[device]);
      edgeWeight[device-1] = benchmarkPCIBus(s, device-1);
      n->getChild(device-1).setEdgeWeight(edgeWeight[device-1]);
    }
  }

  if (NULL == pS) {
    // send the result back to the host
    MPI_Isend(static_cast<void*>(weight), total, MPI_DOUBLE, 0, xWeight,
              MPI_COMM_WORLD, &req[0]);
    MPI_Isend(static_cast<void*>(edgeWeight), total-1, MPI_DOUBLE, 0,
              xEdgeWeight, MPI_COMM_WORLD, &req[1]);
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
  }

  // clean up
  delete [] weight;
  weight = NULL;
  delete [] edgeWeight;
  edgeWeight = NULL;
  if (pS == NULL) {
    delete s;
    s = NULL;
  }
}

// TODO(den4gr)
/*
  takes a subdomain containing results and copies it into original
  buffer, accounting for invalid ghost zone around edges
*/
void copy_result_block(DTYPE* buffer, SubDomain3D* s, int pyramidHeight) { }

void copy_results(DTYPE* buffer, Cluster* cluster, int pyramidHeight) {
  if (NULL == buffer) return;

  /* get work from all parents and children in cluster */
  for (unsigned int n = 0; n < cluster->getNumNodes(); ++n) {
    Node &node = cluster->getNode(n);
    unsigned int num = node.numSubDomains();
    for (unsigned int block =0; block < num; ++block) {
      copy_result_block(buffer, node.getSubDomain(block), pyramidHeight);
    }

    for (unsigned int c = 0; c < node.getNumChildren(); ++c) {
      Node* child = &(node.getChild(c));
      num = child->numSubDomains();

      for (unsigned int block =0; block < num; ++block) {
        copy_result_block(buffer, child->getSubDomain(block), pyramidHeight);
      }
    }
  }
}

bool isSegmentFace(NeighborTag neighbor) {
  return xFace5 >= neighbor && xFace0 <= neighbor;
}

bool isSegmentPole(NeighborTag neighbor) {
  return xPole0 <= neighbor && xPole11 >= neighbor;
}

bool isSegmentCorner(NeighborTag neighbor) {
 return xCorner0 <= neighbor && xCorner7 >= neighbor;
}

void getCornerDimensions(NeighborTag neighbor, int* segmentLength,
                          int* segmentOffset, SubDomain3D* dataBlock,
                          const int kBorder[3]) {
  if (xCorner0 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xCorner1 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
  } else if (xCorner2 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xCorner3 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
  } else if (xCorner4 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xCorner5 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
  } else if (xCorner6 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xCorner7 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
  }
}

void getFaceDimensions(NeighborTag neighbor, int* segmentLength,
    int* segmentOffset, SubDomain3D* dataBlock,
    const int kBorder[3]) {
  if (xFace0 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xFace1 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xFace2 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xFace3 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xFace4 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xFace5 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
  }
}

void getPoleDimensions(NeighborTag neighbor, int* segmentLength,
                          int* segmentOffset, SubDomain3D* dataBlock,
                          const int kBorder[3]) {
  if (xPole0 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xPole1 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xPole2 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xPole3 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xPole4 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xPole5 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
  } else if (xPole6 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
  } else if (xPole7 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xPole8 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = kBorder[2];
  } else if (xPole9 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = kBorder[1];
    segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
  } else if (xPole10 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
  } else if (xPole11 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    segmentOffset[0] = kBorder[0];
    segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentOffset[2] = kBorder[2];
  }
}

// TODO(den4gr)
void getSegmentDimensions(NeighborTag neighbor, int* segmentLength,
    int* segmentOffset, SubDomain3D* dataBlock,
    const int kBorder[3]) {
  int blockLength[3] = {  dataBlock->getLength(0),
    dataBlock->getLength(1),
    dataBlock->getLength(2) };
  if (isSegmentFace(neighbor)) {
    getFaceDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
                      kBorder);
  } else if (isSegmentPole(neighbor)) {
    getPoleDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
                      kBorder);
  } else if (isSegmentCorner(neighbor)) {
    getCornerDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
                        kBorder);
  }
}

//TODO(den4gr)
/* copies to/from the buffer based on the bool flag */
void copySegment(NeighborTag neighbor, SubDomain3D* dataBlock,
    DTYPE* sendBuffer, const int kBorder[3], const bool kCopyToBuffer) {
  int segmentLength[3] = { 0 };
  int segmentOffset[3] = { 0 };
  int blockLength[3] = {  dataBlock->getLength(0),
                          dataBlock->getLength(1),
                          dataBlock->getLength(2) };
  getSegmentDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
      kBorder);
  for (int i = 0; i < segmentLength[0]; ++i) {
    for (int j = 0; j < segmentLength[1]; ++j) {
      for (int k = 0; k < segmentLength[2]; ++k) {
        int bufferIndex = i * segmentLength[1] * segmentLength[2] +
                        j * segmentLength[2] +
                        k;
        int blockI = i + segmentOffset[0];
        int blockJ = j + segmentOffset[1];
        int blockK = k + segmentOffset[2];
        int blockIndex =  blockI * blockLength[1] * blockLength[2] +
                        blockJ * blockLength[2] +
                        blockK;
        if (kCopyToBuffer)
          sendBuffer[bufferIndex] = dataBlock->getBuffer()[blockIndex];
        else
          dataBlock->getBuffer()[blockIndex] = sendBuffer[bufferIndex];
      }
    }
  }
}

//TODO(den4gr)
void exchangeSegments(NeighborTag neighbor, SubDomain3D* dataBlock,
                      DTYPE* sendBuffer, DTYPE* receiveBuffer) {
  /* non-blocking send buffer to node */

  /* blocking receive buffer from node */

  /* wait for send to finish */
}


/*
   TODO(den4gr)
 */
void updateAllStaleData(Node* node, const int kPyramidHeight) {

}

/*
   TODO(den4gr)
 */
void updateStaleBlockData(SubDomain3D* dataBlock, const int kBorder[3]) {
  const int kDepth  = dataBlock->getLength(0) - 2 * kBorder[0];
  const int kWidth  = dataBlock->getLength(1) - 2 * kBorder[1];
  const int kHeight = dataBlock->getLength(2) - 2 * kBorder[2];
  const int kNumAdjacentBlocks = 26;
  int maxSegmentSize = 0, maxPoleSize = 0;
  DTYPE* sendBuffer = NULL;
  DTYPE* receiveBuffer = NULL;
  /* check size of six faces */
  maxSegmentSize = max(kBorder[0] * kWidth * kHeight,
      max(kBorder[2] * kDepth * kHeight,
        kBorder[1] * kDepth * kWidth));
  /* check size of 12 poles */
  maxPoleSize = max(kBorder[0] * kBorder[1] * kWidth,
      max(kBorder[1] * kBorder[2] * kDepth,
        kBorder[0] * kBorder[2] * kHeight));
  maxSegmentSize = max(maxSegmentSize, maxPoleSize);
  /* check size of 8 corners */
  maxSegmentSize = max(maxSegmentSize, kBorder[0] * kBorder[1] * kBorder[2]);
  /* create send and receive buffer that are large enough for
     largest halo segment */
  sendBuffer = new DTYPE[maxSegmentSize];
  receiveBuffer = new DTYPE[maxSegmentSize];

  /* LOOP: over all halo segments */
  for (NeighborTag neighbor = xNeighborBegin;
      neighbor < xNeighborEnd;
      ++neighbor) {
    /* copy halo segment to buffer */
    bool copyBlockToBuffer = true;
    copySegment(neighbor, dataBlock, sendBuffer, kBorder, copyBlockToBuffer);
    exchangeSegments(neighbor, dataBlock, sendBuffer, receiveBuffer);
    /* copy receive buffer into subdomain */
    copyBlockToBuffer = false;
    copySegment(neighbor, dataBlock, sendBuffer, kBorder, copyBlockToBuffer);
  }
  delete [] sendBuffer;
  sendBuffer = NULL;
  delete [] receiveBuffer;
  receiveBuffer = NULL;
}

void processCPUWork(Node* machine, const int kPyramidHeight, const int kBornMin,
    const int kBornMax, const int kDieMin, const int kDieMax) {
  for (unsigned int task = 0; task < machine->numSubDomains(); ++task) {
    processSubDomain(kCPUIndex, machine->getSubDomain(task), kPyramidHeight,
        kBornMin, kBornMax, kDieMin, kDieMax);
  }
}

void processGPUWork(Node* machine, const int kPyramidHeight, const int kBornMin,
    const int kBornMax, const int kDieMin, const int kDieMax) {
  for (unsigned int gpuIndex = 0;
      gpuIndex < machine->getNumChildren();
      ++gpuIndex) {
    Node* currentDevice = &(machine->getChild(gpuIndex));
    for (unsigned int task = 0;
        task < currentDevice->numSubDomains();
        ++task) {
      processSubDomain(gpuIndex, currentDevice->getSubDomain(task),
          kPyramidHeight, kBornMin, kBornMax, kDieMin, kDieMax);
    }
  }
}

void processWork(Node* machine, const int kIterations, const int kPyramidHeight,
    const int kBornMin, const int kBornMax, const int kDieMin,
    const int kDieMax) {
  int currentPyramidHeight = kPyramidHeight;
  const int kFirstIteration = 0;
  for (int iter = kFirstIteration; iter < kIterations; iter += kPyramidHeight) {
    if (iter + kPyramidHeight > kIterations) {
      currentPyramidHeight = kIterations - iter;
    }
    /* The data is initially sent with the ghost zones */
    if (kFirstIteration < iter) { updateAllStaleData(machine, kPyramidHeight); }
    processCPUWork(machine, currentPyramidHeight, kBornMin, kBornMax,
        kDieMin, kDieMax);
    processGPUWork(machine, currentPyramidHeight, kBornMin, kBornMax,
        kDieMin, kDieMax);
  }
}

void getResultsFromCluster(Cluster* cluster) {
  /* TODO(den4gr) receives results, needs to be asynchronous */
  const bool kNoInterleavedCompute = false;
  for (int nodeRank = 1; nodeRank < cluster->getNumNodes(); ++nodeRank) {
    receiveData(nodeRank, &(cluster->getNode(nodeRank)), kNoInterleavedCompute);
  }
}

void sendWorkToCluster(Cluster* cluster) {
  /* TODO(den4gr) needs to be parallel.
      send the work to each node. */
  for (unsigned int node = 1; node < cluster->getNumNodes(); ++node) {
    sendData(&(cluster->getNode(node)));
  }
}

void benchmarkCluster(Cluster* cluster, SubDomain3D* data,
                      const int kIterations,
                      const int kBornMin, const int kBornMax,
                      const int kDieMin, const int kDieMax) {
  /* TODO(den4gr) this is inefficient, need to use Bcast */
  for (unsigned int node = 1; node < cluster->getNumNodes(); ++node) {
    benchmarkNode(&(cluster->getNode(node)), data);
  }
  benchmarkMyself(&(cluster->getNode(0)), data, kIterations, kBornMin, kBornMax,
                  kDieMin, kDieMax);
}

void runDistributedCell(int rank, int numTasks, DTYPE *data, int x_max,
    int y_max, int z_max, int iterations, int bornMin,
    int bornMax, int dieMin, int dieMax) {
  // hack because we want the compiler to give us the
  // stencil size, but we don't want to have to include
  // the cuda headers in every file, so we convert
  // it to an int array for the time-being.
  dim3 stencil_size(1, 1, 1);
  int new_stencil_size[3] = {stencil_size.z, stencil_size.y, stencil_size.x};
  int deviceCount = 0;
  const int kPyramidHeight = 1;
  Node myWork;
  Cluster* cluster = NULL;
  struct timeval rec_start, rec_end, comp_start, comp_end, process_start,
                 process_end, balance_start, balance_end;

  myWork.setRank(rank);
  getNumberOfChildren(&deviceCount);
  myWork.setNumChildren(deviceCount);
  if (0 == rank) {
    Decomposition decomp;
    Balancer lb;
    double balance_sec = -1.0, time_root_compute = -1.0;
    double time_root_receive = -1.0, total_sec = -1.0;
    // get the number of children from other nodes
    cluster = new Cluster(numTasks);
    cluster->getNode(0).setNumChildren(deviceCount);
    receiveNumberOfChildren(numTasks, cluster);
    /* perform domain decomposition */
    int numElements[3] = {z_max, y_max, x_max};
    decomp.decompose(data, 3, numElements, new_stencil_size, PYRAMID_HEIGHT);
#ifdef DEBUG
    printDecomposition(decomp);
#endif
    benchmarkCluster(cluster, decomp.getSubDomain(0), iterations,
                      bornMin, bornMax, dieMin, dieMax);
    /* now perform the load balancing, assigning task blocks to each node */
    gettimeofday(&balance_start, NULL);
    // passing a 0 means use cpu and gpu on all nodes
    lb.perfBalance(*cluster, decomp, 0);
    // lb.balance(*cluster, decomp, 0);
    gettimeofday(&balance_end, NULL);
    printCluster(*cluster);  // DEBUG
    balance_sec = secondsElapsed(balance_start, balance_end);
    fprintf(stderr, "***********\nBALANCE TIME: %f seconds.\n", balance_sec);
    gettimeofday(&process_start, NULL);
    sendWorkToCluster(cluster);
    // TODO(den4gr) Is this a deep copy??
    // root's work is in the first node
    myWork = cluster->getNode(0);
    /* PROCESS ROOT NODE WORK */
    gettimeofday(&comp_start, NULL);
    processWork(&myWork, iterations, kPyramidHeight, bornMin, bornMax,
        dieMin, dieMax);
    gettimeofday(&comp_end, NULL);
    time_root_compute= secondsElapsed(comp_start, comp_end);
    fprintf(stdout, "*********\nroot processing time: %f sec\n",
        time_root_compute);

    gettimeofday(&rec_start, NULL);
    getResultsFromCluster(cluster);
    gettimeofday(&rec_end, NULL);

    gettimeofday(&process_end, NULL);
    time_root_receive = secondsElapsed(rec_start, rec_end);
    fprintf(stdout, "***********\nroot receive time: %f sec\n",
        time_root_receive);
    total_sec = secondsElapsed(process_start, process_end);
    fprintf(stdout, "***********\nTOTAL TIME: %f.\n", total_sec);

    delete cluster;
    cluster = NULL;
  } else {
    const bool kInterleaveProcessing = true;
    int remainingIterations = 0;
    // send number of children to root
    sendNumberOfChildren(0, deviceCount);
    benchmarkMyself(&myWork, NULL, iterations, bornMin, bornMax,
        dieMin, dieMax);
    receiveData(0, &myWork, kInterleaveProcessing, kPyramidHeight,
                bornMin, bornMax, dieMin, dieMax);
    remainingIterations = iterations - kPyramidHeight;
    processWork(&myWork, remainingIterations, kPyramidHeight, bornMin, bornMax,
                dieMin, dieMax);
    // send my work back to the root
    myWork.setRank(0);
    sendData(&myWork);
  }
}
