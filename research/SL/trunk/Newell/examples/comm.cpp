#include "comm.h"
#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <boost/scoped_array.hpp>
static const int kNumNeighbors3D = 26;
MPI_Request* mpi_requests = NULL;
static double buffer_copy_time = 0.0;
static MPI_Datatype mpi_types[kNumNeighbors3D];
static MPI_Datatype mpi_sub_types[kNumNeighbors3D];
static int segment_offsets[kNumNeighbors3D][kSendAndReceive];
static const int kSendIndex = 0;
static const int kReceiveIndex = 1;

using namespace boost;

double secondsElapsed(struct timeval start, struct timeval stop) {
  return static_cast<double> ((stop.tv_sec - start.tv_sec) +
          (stop.tv_usec - start.tv_usec) / 1000000.0);
}

void sendDataToNode(const int kDestRank, int device, SubDomain* block) {
  // first send number of dimensions
  MPI_Request req;
  const int kNodeBufferSize = 40;
  int subdomain_envelope[40] = {0};
  const int kNumNeighbors3D = 26;
  memcpy(&subdomain_envelope[xID0], block->getId(), 3 * sizeof (int));
  memcpy(&subdomain_envelope[xGridDimension0], block->getGridDim(),
          3 * sizeof (int));
  subdomain_envelope[xDeviceIndex] = device;
  for (int i = 0; i < 3; ++i) {
    if (block->getLength(i) > 0)
      ++subdomain_envelope[xNumberDimensions];
  }
  subdomain_envelope[xLength0] = block->getLength(0);
  subdomain_envelope[xLength1] = block->getLength(1);
  subdomain_envelope[xLength2] = block->getLength(2);
  subdomain_envelope[xOffset0] = block->getOffset(0);
  subdomain_envelope[xOffset1] = block->getOffset(1);
  subdomain_envelope[xOffset2] = block->getOffset(2);
  memcpy(&subdomain_envelope[xNeighborsStartIndex], block->getNeighbors(),
          kNumNeighbors3D * sizeof (int));
  int result = MPI_Isend(static_cast<void*> (subdomain_envelope),
          kNodeBufferSize, MPI_INT, kDestRank, xSubDomainEnvelope,
          MPI_COMM_WORLD, &req);
  mpiCheckError(result);
  // third send data
  // first we have to stage the data into contiguous memory
  int total_size = 1;
  for (int i = 0; i < subdomain_envelope[xNumberDimensions]; ++i)
    total_size *= block->getLength(i);
  result = MPI_Send(static_cast<void*> (block->getBuffer()), total_size,
          SL_MPI_TYPE, kDestRank, xData, MPI_COMM_WORLD);
  mpiCheckError(result);
  result = MPI_Wait(&req, MPI_STATUSES_IGNORE);
  mpiCheckError(result);
}

void receiveNumberOfChildren(const int kNumTasks, Cluster* cluster) {
  scoped_array<MPI_Request> reqs(new MPI_Request[kNumTasks - 1]);
  scoped_array<int> num_children(new int[kNumTasks - 1]());

  for (int i = 0; i < kNumTasks - 1; i++) {
    // receive next count
    int result = MPI_Irecv(static_cast<void*> (&(num_children[i])), 1, MPI_INT,
            i + 1, xChildren, MPI_COMM_WORLD, &(reqs[i]));
    mpiCheckError(result);
  }
  int result = MPI_Waitall(kNumTasks - 1, reqs.get(), MPI_STATUSES_IGNORE);
  mpiCheckError(result);
  for (int task = 0; task < kNumTasks - 1; task++) {
    cluster->getNode(task + 1).setNumChildren(num_children[task]);
  }
}

void sendNumberOfChildren(const int kDestinationRank, const int kNumChildren) {
  int sendNumChildrenBuffer = kNumChildren;
  int my_rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int result = MPI_Send(static_cast<void*> (&sendNumChildrenBuffer), 1,
          MPI_INT, kDestinationRank, xChildren, MPI_COMM_WORLD);
  mpiCheckError(result);
}

void sendData(Node* node) {
  // count how many task blocks, total, are going to be sent
  int total_blocks = node->numTotalSubDomains();

  // send node number of blocks
  MPI_Request request;
  int node_rank = node->getRank();
  int result = MPI_Isend(static_cast<void*> (&total_blocks), 1, MPI_INT,
          node->getRank(), xNumBlocks, MPI_COMM_WORLD, &request);
  mpiCheckError(result);
  const int kCpuIndex = -1;
  for (unsigned int block_index = 0;
          block_index < node->numSubDomains();
          ++block_index) {
    sendDataToNode(node_rank, kCpuIndex, node->getSubDomain(block_index));
  }
  for (unsigned int gpu_index = 0;
          gpu_index < node->getNumChildren();
          ++gpu_index) {
    Node& gpu = node->getChild(gpu_index);
    for (unsigned int block_index = 0;
            block_index < gpu.numSubDomains();
            ++block_index) {
      sendDataToNode(node_rank, gpu_index, gpu.getSubDomain(block_index));
    }
  }
  // wait for first send to finish
  result = MPI_Waitall(1, &request, MPI_STATUSES_IGNORE);
  mpiCheckError(result);
}

void benchmarkNode(Node* node, SubDomain* benchmark_block) {
  struct timeval start, end;
  double total_sec = 0.0;
  MPI_Status status;
  const int kCommIterations = 3;
  /*{
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof (hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  } // */
  {
    int tmp_device = -1;
    SubDomain * blocks[] = {NULL, NULL, NULL};
    gettimeofday(&start, NULL);
    for (int i = 0; i < kCommIterations; ++i) {
      // send task block to every device on that node
      sendDataToNode(node->getRank(), -1, benchmark_block);
      blocks[i] = receiveDataFromNode(node->getRank(), &tmp_device);
    }
    gettimeofday(&end, NULL);
    for (int i = 0; i < kCommIterations; ++i)
      delete blocks[i];
  }

  total_sec = secondsElapsed(start, end) / kCommIterations;
  // how fast is the connection between root and child nodes
  // multiply by 2 to account for there and back
  node->setEdgeWeight(1 / (2 * total_sec));
  // receive results for each device
  unsigned int total = node->getNumChildren() + 1;

  // use one buffer for both to reduce # MPI messages
  scoped_array<double> task_per_sec(new double[2 * total - 1]());
  double *edge_weight = &task_per_sec[total];

  int result = MPI_Recv(static_cast<void*> (task_per_sec.get()), 2 * total - 1,
          MPI_DOUBLE, node->getRank(), xWeight, MPI_COMM_WORLD, &status);
  mpiCheckError(result);

  // set the appropriate fields in the node and its children
  for (unsigned int device = 0; device < total; ++device) {
    double weight = task_per_sec[device];
    if (device == 0) {
      // the first weight is for the cpu
      node->setWeight(weight);
    } else {
      double edgeWeight = edge_weight[device - 1];
      node->getChild(device - 1).setWeight(weight);
      node->getChild(device - 1).setEdgeWeight(edgeWeight);
    }
  }
}

/* output variables: buf, size */
SubDomain* receiveDataFromNode(int rank, int* device) {
  int id[3] = {-1, -1, -1};
  int my_rank = -1, result = 0, size = 0;
  const int kSubDomainEnvelopeSize = 40;
  int subdomain_envelope[40] = {0};
  SubDomain *sub_domain = NULL;
  MPI_Status status;

  result = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  mpiCheckError(result);

  // receive dimensionality of data
  result = MPI_Recv(static_cast<void*> (subdomain_envelope),
          kSubDomainEnvelopeSize, MPI_INT, rank, xSubDomainEnvelope,
          MPI_COMM_WORLD, &status);
  mpiCheckError(result);
  // receive size of data
  *device = subdomain_envelope[xDeviceIndex];
  memcpy(id, &subdomain_envelope[xID0], 3 * sizeof (int));
  if (2 == subdomain_envelope[xNumberDimensions]) {
    sub_domain = new SubDomain(&subdomain_envelope[xID0],
            subdomain_envelope[xOffset0],
            subdomain_envelope[xLength0],
            subdomain_envelope[xOffset1],
            subdomain_envelope[xLength1],
            subdomain_envelope[xGridDimension0],
            subdomain_envelope[xGridDimension1],
            &subdomain_envelope[xNeighborsStartIndex]);
  } else if (3 == subdomain_envelope[xNumberDimensions]) {
    sub_domain = new SubDomain(&subdomain_envelope[xID0],
            subdomain_envelope[xOffset0],
            subdomain_envelope[xLength0],
            subdomain_envelope[xOffset1],
            subdomain_envelope[xLength1],
            subdomain_envelope[xOffset2],
            subdomain_envelope[xLength2],
            subdomain_envelope[xGridDimension0],
            subdomain_envelope[xGridDimension1],
            subdomain_envelope[xGridDimension2],
            &subdomain_envelope[xNeighborsStartIndex]);
  }

  size = subdomain_envelope[xLength0];
  for (int i = 1; i < subdomain_envelope[xNumberDimensions]; ++i)
    size *= subdomain_envelope[xLength0 + i];

  result = MPI_Recv(static_cast<void*> (sub_domain->getBuffer()), size,
          SL_MPI_TYPE, rank, xData, MPI_COMM_WORLD, &status);
  mpiCheckError(result);
  return sub_domain;
}

bool isSegmentFace(NeighborTag3D neighbor) {
  return x3DFace5 >= neighbor && x3DFace0 <= neighbor;
}

bool isSegmentPole(NeighborTag3D neighbor) {
  return x3DPole0 <= neighbor && x3DPole11 >= neighbor;
}

bool isSegmentPole(NeighborTag2D neighbor) {
  return x2DPole0 <= neighbor && x2DPole3 >= neighbor;
}

bool isSegmentCorner(NeighborTag3D neighbor) {
  return x3DCorner0 <= neighbor && x3DCorner7 >= neighbor;
}

bool isSegmentCorner(NeighborTag2D neighbor) {
  return x2DCorner0 <= neighbor && x2DCorner3 >= neighbor;
}

void getCornerDimensions(NeighborTag3D neighbor, int* segmentLength,
        int* segmentOffset, SubDomain* dataBlock,
        const int kBorder[3], const bool kBlockToBuffer) {
  if (x3DCorner0 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = 0;
      segmentOffset[2] = 0;
    }
  } else if (x3DCorner1 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = 0;
      segmentOffset[2] = dataBlock->getLength(2) - kBorder[2];
    }
  } else if (x3DCorner2 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = dataBlock->getLength(1) - kBorder[1];
      segmentOffset[2] = 0;
    }
  } else if (x3DCorner3 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = dataBlock->getLength(1) - kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - kBorder[2];
    }
  } else if (x3DCorner4 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = dataBlock->getLength(0) - kBorder[0];
      segmentOffset[1] = 0;
      segmentOffset[2] = 0;
    }
  } else if (x3DCorner5 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    } else {
      segmentOffset[0] = dataBlock->getLength(0) - kBorder[0];
      segmentOffset[1] = 0;
      segmentOffset[2] = dataBlock->getLength(2) - kBorder[2];
    }
  } else if (x3DCorner6 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = dataBlock->getLength(0) - kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - kBorder[1];
      segmentOffset[2] = 0;
    }
  } else if (x3DCorner7 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    } else {
      segmentOffset[0] = dataBlock->getLength(0) - kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - kBorder[2];
    }
  }
}

void getCornerDimensions(NeighborTag2D neighbor, int* segmentLength,
        int* segmentOffset, SubDomain* dataBlock,
        const int kBorder[3], const bool kBlockToBuffer) {
  const int kHeight = dataBlock->getLength(0);
  const int kWidth = dataBlock->getLength(1);
  segmentLength[0] = kBorder[0];
  segmentLength[1] = kBorder[1];
  if (x2DCorner0 == neighbor) {
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = 0;
    }
  } else if (x2DCorner1 == neighbor) {
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kWidth - 2 * kBorder[1];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = kWidth - kBorder[1];
    }
  } else if (x2DCorner2 == neighbor) {
    if (kBlockToBuffer) {
      segmentOffset[0] = kHeight - 2 * kBorder[0];
      segmentOffset[1] = kWidth - 2 * kBorder[1];
    } else {
      segmentOffset[0] = kHeight - kBorder[0];
      segmentOffset[1] = kWidth - kBorder[1];
    }
  } else if (x2DCorner3 == neighbor) {
    if (kBlockToBuffer) {
      segmentOffset[0] = kHeight - 2 * kBorder[0];
      segmentOffset[1] = kBorder[1];
    } else {
      segmentOffset[0] = kHeight - kBorder[0];
      segmentOffset[1] = 0;
    }
  }
}

void getFaceDimensions(NeighborTag3D neighbor, int* segmentLength,
        int* segmentOffset, SubDomain* dataBlock,
        const int kBorder[3], const bool kBlockToMPIBuffer) {
  if (x3DFace0 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    if (kBlockToMPIBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    }
  } else if (x3DFace1 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    if (kBlockToMPIBuffer) {
      segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = dataBlock->getLength(0) - kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    }
  } else if (x3DFace2 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    if (kBlockToMPIBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = 0;
      segmentOffset[2] = kBorder[2];
    }
  } else if (x3DFace3 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    if (kBlockToMPIBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - kBorder[1];
      segmentOffset[2] = kBorder[2];
    }
  } else if (x3DFace4 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToMPIBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = 0;
    }
  } else if (x3DFace5 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToMPIBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - kBorder[2];
    }
  }
}

void getPoleDimensions(NeighborTag3D neighbor, int* segmentLength,
        int* segmentOffset, SubDomain* dataBlock,
        const int kBorder[3], const bool kBlockToBuffer) {
  if (x3DPole0 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = 0;
      segmentOffset[2] = kBorder[2];
    }
  } else if (x3DPole1 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = dataBlock->getLength(1) - kBorder[1];
      segmentOffset[2] = kBorder[2];
    }
  } else if (x3DPole2 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = dataBlock->getLength(0) - kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - kBorder[1];
      segmentOffset[2] = kBorder[2];
    }
  } else if (x3DPole3 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = dataBlock->getLength(0) - kBorder[0];
      segmentOffset[1] = 0;
      segmentOffset[2] = kBorder[2];
    }
  } else if (x3DPole4 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = 0;
    }
  } else if (x3DPole5 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - kBorder[2];
    }
  } else if (x3DPole6 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    } else {
      segmentOffset[0] = dataBlock->getLength(0) - kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - kBorder[2];
    }
  } else if (x3DPole7 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = dataBlock->getLength(0) - 2 * kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = dataBlock->getLength(0) - kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = 0;
    }
  } else if (x3DPole8 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = 0;
      segmentOffset[2] = 0;
    }
  } else if (x3DPole9 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = 0;
      segmentOffset[2] = dataBlock->getLength(2) - kBorder[2];
    }
  } else if (x3DPole10 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - kBorder[1];
      segmentOffset[2] = dataBlock->getLength(2) - kBorder[2];
    }
  } else if (x3DPole11 == neighbor) {
    segmentLength[0] = dataBlock->getLength(0) - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    segmentLength[2] = kBorder[2];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - 2 * kBorder[1];
      segmentOffset[2] = kBorder[2];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = dataBlock->getLength(1) - kBorder[1];
      segmentOffset[2] = 0;
    }
  }
}

void getPoleDimensions(NeighborTag2D neighbor, int* segmentLength,
        int* segmentOffset, SubDomain* dataBlock,
        const int kBorder[3], const bool kBlockToMPIBuffer) {
  const int kHeight = dataBlock->getLength(0);
  const int kWidth = dataBlock->getLength(1);
  if (x2DPole0 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kWidth - 2 * kBorder[1];
    if (kBlockToMPIBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = kBorder[1];
    }
  } else if (x2DPole1 == neighbor) {
    segmentLength[0] = kHeight - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    if (kBlockToMPIBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kWidth - 2 * kBorder[1];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kWidth - kBorder[1];
    }
  } else if (x2DPole2 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kWidth - 2 * kBorder[1];
    if (kBlockToMPIBuffer) {
      segmentOffset[0] = kHeight - 2 * kBorder[0];
      segmentOffset[1] = kBorder[1];
    } else {
      segmentOffset[0] = kHeight - kBorder[0];
      segmentOffset[1] = kBorder[1];
    }
  } else if (x2DPole3 == neighbor) {
    segmentLength[0] = kHeight - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    if (kBlockToMPIBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = 0;
    }
  }
}

/**
  brief: creates the MPI Datatype for a Stencil Ghost Zone segment.
 */
void initGhostZoneType(const SubDomain* kBlock, NeighborTag3D neighbor,
        const int kBorder[], MPI_Datatype* type, MPI_Datatype* sub_type,
        int ghost_offset[]) {
  int length[] = {0, 0, 0};
  int send_offset[] = {0, 0, 0};
  int receive_offset[] = {0, 0, 0};
  MPI_Aint size_of_sl_mpi_type = 0;
  int segment_height = 1, segment_width = 1, segment_depth = 1;
  int block_height = 1, block_width = 1, block_depth = 1;
  MPI_Datatype one_dimension, two_dimension;
  bool block_to_mpi_buffer = false;
  getSegmentDimensions(neighbor, length, receive_offset,
          const_cast<SubDomain*> (kBlock), kBorder, block_to_mpi_buffer);
  block_to_mpi_buffer = true;
  getSegmentDimensions(neighbor, length, send_offset,
          const_cast<SubDomain*> (kBlock), kBorder, block_to_mpi_buffer);
  segment_depth = length[0];
  segment_height = length[1];
  segment_width = length[2];
  block_depth = kBlock->getLength(0);
  block_height = kBlock->getLength(1);
  block_width = kBlock->getLength(2);

  // set the offsets for the beginning of the blocks for send and receive
  ghost_offset[kReceiveIndex] = receive_offset[0] * block_height * block_width +
          receive_offset[1] * block_width +
          receive_offset[2];
  ghost_offset[kSendIndex] = send_offset[0] * block_height * block_width +
          send_offset[1] * block_width +
          send_offset[2];
  MPI_Type_extent(SL_MPI_TYPE, &size_of_sl_mpi_type);
  MPI_Type_vector(1, segment_width, block_width, SL_MPI_TYPE,
          &one_dimension);
  MPI_Type_hvector(segment_height, 1, block_width * size_of_sl_mpi_type,
          one_dimension, &two_dimension);
  MPI_Type_hvector(segment_depth, 1,
          block_width * block_height * size_of_sl_mpi_type, two_dimension,
          type);
  MPI_Type_commit(type);
}

void initGhostZoneType(const SubDomain* kBlock, NeighborTag2D neighbor,
        const int kBorder[], MPI_Datatype* type, MPI_Datatype* sub_type,
        int ghost_offset[]) {
  int length[] = {0, 0, 0};
  int send_offset[] = {0, 0, 0};
  int receive_offset[] = {0, 0, 0};
  int segment_height = 1, segment_width = 1, segment_depth = 1;
  int block_height = 1, block_width = 1, block_depth = 1;
  bool block_to_mpi_buffer = false;
  getSegmentDimensions(neighbor, length, receive_offset,
          const_cast<SubDomain*> (kBlock), kBorder, block_to_mpi_buffer);
  block_to_mpi_buffer = true;
  getSegmentDimensions(neighbor, length, send_offset,
          const_cast<SubDomain*> (kBlock), kBorder, block_to_mpi_buffer);
  segment_height = length[0];
  segment_width = length[1];
  block_height = kBlock->getLength(0);
  block_width = kBlock->getLength(1);

  ghost_offset[kReceiveIndex] = receive_offset[0] * block_width +
          receive_offset[1];
  ghost_offset[kSendIndex] = send_offset[0] * block_width +
          send_offset[1];
  MPI_Type_vector(segment_height, segment_width, block_width, SL_MPI_TYPE,
          sub_type);
  MPI_Type_commit(sub_type);
  *type = *sub_type;
}

void initAll2DGhostZoneTypes(const SubDomain* kBlock, const int kBorder[],
        MPI_Datatype types[], MPI_Datatype sub_types[],
        int offsets[][kSendAndReceive]) {
  for (NeighborTag2D neighbor = x2DNeighborBegin;
          neighbor < x2DNeighborEnd;
          ++neighbor) {
    initGhostZoneType(kBlock, neighbor, kBorder, &types[neighbor],
            &sub_types[neighbor], offsets[neighbor]);
  }
}

void initAll3DGhostZoneTypes(const SubDomain* kBlock, const int kBorder[],
        MPI_Datatype types[], MPI_Datatype sub_types[],
        int offsets[][kSendAndReceive]) {
  for (NeighborTag3D neighbor = x3DNeighborBegin;
          neighbor < x3DNeighborEnd;
          ++neighbor) {
    initGhostZoneType(kBlock, neighbor, kBorder, &types[neighbor],
            &sub_types[neighbor], offsets[neighbor]);
  }
}

void initAllGhostZoneTypes(const SubDomain* kBlock, const int kNumberDimensions,
        const int kBorder[], MPI_Datatype types[], MPI_Datatype sub_types[],
        int offsets[][kSendAndReceive]) {
  const int k2D = 2;
  const int k3D = 3;
  if (k2D == kNumberDimensions) {
    initAll2DGhostZoneTypes(kBlock, kBorder, types, sub_types, offsets);
  } else if (k3D == kNumberDimensions) {
    initAll3DGhostZoneTypes(kBlock, kBorder, types, sub_types, offsets);
  }
}

int getMPITagForSegment(NeighborTag2D segment) {
  return static_cast<MPITagType> (static_cast<int> (segment) + 20);
}

int getMPITagForSegment(NeighborTag3D segment) {
  return static_cast<MPITagType> (static_cast<int> (segment) + 30);
}

int getMPITagForSegmentData(NeighborTag2D segment, const int kBlockIndex) {
  return static_cast<MPITagType> (static_cast<int> (segment) * kMagicNumber
          + kBlockIndex);
}

int getMPITagForSegmentData(NeighborTag3D segment, const int kBlockIndex) {
  return static_cast<MPITagType> (static_cast<int> (segment) * kMagicNumber
          + kBlockIndex);
}
// TODO(den4gr)

void getSegmentDimensions(NeighborTag3D neighbor, int* segmentLength,
        int* segmentOffset, SubDomain* dataBlock,
        const int kBorder[3], const bool kBlockToMPIBuffer) {
  if (isSegmentFace(neighbor)) {
    getFaceDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
            kBorder, kBlockToMPIBuffer);
  } else if (isSegmentPole(neighbor)) {
    getPoleDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
            kBorder, kBlockToMPIBuffer);
  } else if (isSegmentCorner(neighbor)) {
    getCornerDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
            kBorder, kBlockToMPIBuffer);
  }
}

void getSegmentDimensions(NeighborTag2D neighbor, int* segmentLength,
        int* segmentOffset, SubDomain* dataBlock,
        const int kBorder[3], const bool kBlockToMPIBuffer) {
  if (isSegmentPole(neighbor)) {
    getPoleDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
            kBorder, kBlockToMPIBuffer);
  } else if (isSegmentCorner(neighbor)) {
    getCornerDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
            kBorder, kBlockToMPIBuffer);
  }
}

//TODO(den4gr)

/* copies to another block on the current node, to avoid the cost of MPI comm */
void copySegment(NeighborTag3D neighbor, Node* node, SubDomain* source_block,
        const int kBorder[3]) {
  bool segment_in_ghost_zone = false;
  const int kNeighborBlockIndex = source_block->getNeighborIndex(neighbor);
  SubDomain* destination_block = node->getSubDomainLinear(kNeighborBlockIndex);
  int source_segment_length[3] = {0};
  int source_segment_offset[3] = {0};
  int destination_segment_offset[3] = {0};
  int block_length[3] = {source_block->getLength(0),
    source_block->getLength(1),
    source_block->getLength(2)};
  DTYPE* source_buffer = source_block->getBuffer();
  DTYPE* destination_buffer = destination_block->getBuffer();

  getSegmentDimensions(neighbor, source_segment_length, source_segment_offset,
          source_block, kBorder, segment_in_ghost_zone);
  const NeighborTag3D kOppositeNeighbor = getOppositeNeighbor3D(neighbor);
  segment_in_ghost_zone = true;
  getSegmentDimensions(kOppositeNeighbor, source_segment_length,
          destination_segment_offset, source_block, kBorder,
          segment_in_ghost_zone);

  for (int i = 0; i < source_segment_length[0]; ++i) {
    for (int j = 0; j < source_segment_length[1]; ++j) {
      for (int k = 0; k < source_segment_length[2]; ++k) {
        int source_I = i + source_segment_offset[0];
        int source_J = j + source_segment_offset[1];
        int source_K = k + source_segment_offset[2];
        int source_index = source_I * block_length[1] * block_length[2] +
                source_J * block_length[2] +
                source_K;
        int destination_I = i + destination_segment_offset[0];
        int destination_J = j + destination_segment_offset[1];
        int destination_K = k + destination_segment_offset[2];
        int destination_index =
                destination_I * block_length[1] * block_length[2] +
                destination_J * block_length[2] +
                destination_K;
        destination_buffer[destination_index] = source_buffer[source_index];
      }
    }
  }
}

//TODO(den4gr)

/* copies to another block on the current node, to avoid the cost of MPI comm */
void copySegment(NeighborTag2D neighbor, Node* node, SubDomain* source_block,
        const int kBorder[2]) {
  bool segment_in_ghost_zone = false;
  const int kNeighborBlockIndex = source_block->getNeighborIndex(neighbor);
  SubDomain* destination_block = node->getSubDomainLinear(kNeighborBlockIndex);
  int source_segment_length[3] = {0};
  int source_segment_offset[3] = {0};
  int destination_segment_offset[3] = {0};
  int block_length[3] = {source_block->getLength(0),
    source_block->getLength(1),
    source_block->getLength(2)};
  DTYPE* source_buffer = source_block->getBuffer();
  DTYPE* destination_buffer = destination_block->getBuffer();

  getSegmentDimensions(neighbor, source_segment_length, source_segment_offset,
          source_block, kBorder, segment_in_ghost_zone);
  const NeighborTag2D kOppositeNeighbor = getOppositeNeighbor2D(neighbor);
  segment_in_ghost_zone = true;
  getSegmentDimensions(kOppositeNeighbor, source_segment_length,
          destination_segment_offset, source_block, kBorder,
          segment_in_ghost_zone);

  for (int i = 0; i < source_segment_length[0]; ++i) {
    for (int j = 0; j < source_segment_length[1]; ++j) {
      int source_I = i + source_segment_offset[0];
      int source_J = j + source_segment_offset[1];
      int source_index = source_I * block_length[1] + source_J;
      int destination_I = i + destination_segment_offset[0];
      int destination_J = j + destination_segment_offset[1];
      int destination_index = destination_I * block_length[1] + destination_J;
      destination_buffer[destination_index] = source_buffer[source_index];
    }
  }
}

/* copies to/from the buffer based on the bool flag */
void copySegment(NeighborTag3D neighbor, SubDomain* dataBlock,
        DTYPE* buffer, const int kBorder[3], const bool kBlockToBuffer,
        int* segmentSize) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  int segmentLength[3] = {0};
  int segmentOffset[3] = {0};
  int blockLength[3] = {dataBlock->getLength(0),
    dataBlock->getLength(1),
    dataBlock->getLength(2)};
  getSegmentDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
          kBorder, kBlockToBuffer);
  if (NULL != segmentSize) {
    *segmentSize = segmentLength[0] *
            segmentLength[1] *
            segmentLength[2];
  }
  for (int i = 0; i < segmentLength[0]; ++i) {
    for (int j = 0; j < segmentLength[1]; ++j) {
      for (int k = 0; k < segmentLength[2]; ++k) {
        int bufferIndex = i * segmentLength[1] * segmentLength[2] +
                j * segmentLength[2] +
                k;
        int blockI = i + segmentOffset[0];
        int blockJ = j + segmentOffset[1];
        int blockK = k + segmentOffset[2];
        int blockIndex = blockI * blockLength[1] * blockLength[2] +
                blockJ * blockLength[2] +
                blockK;
        if (kBlockToBuffer)
          buffer[bufferIndex] = dataBlock->getBuffer()[blockIndex];
        else
          dataBlock->getBuffer()[blockIndex] = buffer[bufferIndex];
      }
    }
  }
  gettimeofday(&stop, NULL);
  buffer_copy_time += secondsElapsed(start, stop);
}

/* copies to/from the buffer based on the bool flag */
void copySegment(NeighborTag2D neighbor, SubDomain* dataBlock,
        DTYPE* buffer, const int kBorder[3], const bool kBlockToBuffer,
        int* segmentSize) {
  int segmentLength[3] = {0};
  int segmentOffset[3] = {0};
  int blockLength[3] = {dataBlock->getLength(0),
    dataBlock->getLength(1),
    dataBlock->getLength(2)};
  getSegmentDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
          kBorder, kBlockToBuffer);
  if (NULL != segmentSize) {
    *segmentSize = segmentLength[0] *
            segmentLength[1];
  }
  for (int i = 0; i < segmentLength[0]; ++i) {
    for (int j = 0; j < segmentLength[1]; ++j) {
      int bufferIndex = i * segmentLength[1] + j;
      int blockI = i + segmentOffset[0];
      int blockJ = j + segmentOffset[1];
      int blockIndex = blockI * blockLength[1] + blockJ;
      if (kBlockToBuffer)
        buffer[bufferIndex] = dataBlock->getBuffer()[blockIndex];
      else
        dataBlock->getBuffer()[blockIndex] = buffer[bufferIndex];
    }
  }
}

/* Sends the ghost zone segment to the neighbor who needs it.
    This assumes that all blocks are sending the same neighbor,
    at the same time, with the same size. This allows us to just
    send the data, without any size or type information about the
    segment
 */
bool sendSegment(const NeighborTag3D kNeighbor, const int kDestinationRank,
        SubDomain* data_block, DTYPE* send_buffer, const int kSize,
        MPI_Request* request) {
  const int kNoNeighbor(-1);
  if (kNoNeighbor < kDestinationRank) {
    // calculate the memory address of first element in segment
    void* segment =
            static_cast<void*> (send_buffer +
            segment_offsets[kNeighbor][kSendIndex]);
    int block_index = data_block->getNeighborIndex(kNeighbor);
    const NeighborTag3D kOppositeNeighbor = getOppositeNeighbor3D(kNeighbor);
    int data_tag = getMPITagForSegmentData(kOppositeNeighbor, block_index);

    int result = MPI_Isend(segment, 1, mpi_types[kOppositeNeighbor],
            kDestinationRank, data_tag, MPI_COMM_WORLD, request);
    mpiCheckError(result);
    return true;
  }
  return false;
}

/* Sends the ghost zone segment to the neighbor who needs it.
    This assumes that all blocks are sending the same neighbor,
    at the same time, with the same size. This allows us to just
    send the data, without any size or type information about the
    segment
 */
bool sendSegment(const NeighborTag2D kNeighbor, const int kDestinationRank,
        SubDomain* dataBlock, DTYPE* send_buffer, const int kSize,
        MPI_Request* request) {
  if (-1 < kDestinationRank) {
    // calculate the memory address of first element in segment
    void* segment_pointer =
            static_cast<void*> (send_buffer + segment_offsets[kNeighbor][kSendIndex]);
    int block_index = dataBlock->getNeighborIndex(kNeighbor);
    const NeighborTag2D kOppositeNeighbor = getOppositeNeighbor2D(kNeighbor);
    int data_tag = getMPITagForSegmentData(kOppositeNeighbor, block_index);
    int result = MPI_Isend(segment_pointer, 1, mpi_types[kOppositeNeighbor],
            kDestinationRank, data_tag, MPI_COMM_WORLD, request);
    mpiCheckError(result);
    return true;
  }
  return false;
}

void receiveSegment(Node* node, int* linearIndex) {
  MPI_Status status;
  SubDomain* block = NULL;
  int result = 0, offset = -1, tag = -1, source_rank = -1;
  result = MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  mpiCheckError(result);

  tag = status.MPI_TAG;
  source_rank = status.MPI_SOURCE;
  *linearIndex = tag % kMagicNumber;
  int neighbor_type_index = tag / kMagicNumber;
  offset = segment_offsets[neighbor_type_index][kReceiveIndex];
  block = node->getSubDomainLinear(*linearIndex);
  DTYPE* block_buffer = block->getBuffer();
  void* segment_pointer = static_cast<void*> (block_buffer + offset);
  result = MPI_Recv(segment_pointer, 1, mpi_types[neighbor_type_index],
          source_rank, tag, MPI_COMM_WORLD, &status);
  mpiCheckError(result);
}

/* TODO(den4gr)
    need to create a function that will send a particular neighbor segment
    for all blocks, and return the buffers and the MPI_Requests
 */
void sendNewGhostZones(const NeighborTag3D kNeighbor, Node* node,
        const int kBorder[3], MPI_Request* requests, int* segment_size,
        int* number_messages_sent) {
  const int kNoNeighbor(-1);
  int my_rank(-1), result(0);
  result = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  mpiCheckError(result);

  for (unsigned int block_index = 0;
          block_index < node->numTotalSubDomains();
          ++block_index) {
    SubDomain* data_block = node->globalGetSubDomain(block_index);
    const int kDestinationRank = data_block->getNeighborLoc(kNeighbor);
    if (kDestinationRank == my_rank) { // copy instead of mpi send
      copySegment(kNeighbor, node, data_block, kBorder);
    } else if (kDestinationRank != kNoNeighbor) {
      DTYPE* send_buffer = data_block->getBuffer();
      /* copy halo segment to buffer */
      bool send_occured = sendSegment(kNeighbor, kDestinationRank, data_block,
              send_buffer, *segment_size, &requests[*number_messages_sent]);
      if (send_occured)
        ++(*number_messages_sent);
    }
  }
}

void sendNewGhostZones(const NeighborTag2D kNeighbor, Node* node,
        const int kBorder[3], MPI_Request* requests, int* segmentSize,
        int* numberMessagesSent) {
  const int kSendIndex = 0;
  int my_rank = -1, result = 0;
  result = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  mpiCheckError(result);

  for (unsigned int blockIndex = 0;
          blockIndex < node->numTotalSubDomains();
          ++blockIndex) {
    SubDomain* data_block = node->globalGetSubDomain(blockIndex);
    const int kDestinationRank = data_block->getNeighborLoc(kNeighbor);
    if (kDestinationRank == my_rank) { // copy instead of mpi send
      copySegment(kNeighbor, node, data_block, kBorder);
    } else {
      DTYPE* send_buffer = data_block->getBuffer();
      /* copy halo segment to buffer */
      bool copy_to_mpi_buffer = true;
      bool didSend = sendSegment(kNeighbor, kDestinationRank, data_block,
              send_buffer, *segmentSize, &requests[*numberMessagesSent]);
      if (didSend)
        ++(*numberMessagesSent);
    }
  }
}

/* This attrocious function is used in the exchange of ghost zones.
  It is used when you want to send a particular ghost zone segment
  to the neighboring cell. If you are sending Block A's face 0, to
  block B, then it will be stored in Block B's face 1 segment.
 */
NeighborTag2D getOppositeNeighbor2D(const NeighborTag2D kNeighbor) {
  if (x2DPole0 == kNeighbor)
    return x2DPole2;
  else if (x2DPole1 == kNeighbor)
    return x2DPole3;
  else if (x2DPole2 == kNeighbor)
    return x2DPole0;
  else if (x2DPole3 == kNeighbor)
    return x2DPole1;
  else if (x2DCorner0 == kNeighbor)
    return x2DCorner2;
  else if (x2DCorner1 == kNeighbor)
    return x2DCorner3;
  else if (x2DCorner2 == kNeighbor)
    return x2DCorner0;
  else if (x2DCorner3 == kNeighbor)
    return x2DCorner1;
  else
    return x2DNeighborEnd;
}

/* This attrocious function is used in the exchange of ghost zones.
  It is used when you want to send a particular ghost zone segment
  to the neighboring cell. If you are sending Block A's face 0, to
  block B, then it will be stored in Block B's face 1 segment.
 */
NeighborTag3D getOppositeNeighbor3D(const NeighborTag3D kNeighbor) {
  if (x3DFace0 == kNeighbor)
    return x3DFace1;
  else if (x3DFace1 == kNeighbor)
    return x3DFace0;
  else if (x3DFace2 == kNeighbor)
    return x3DFace3;
  else if (x3DFace3 == kNeighbor)
    return x3DFace2;
  else if (x3DFace4 == kNeighbor)
    return x3DFace5;
  else if (x3DFace5 == kNeighbor)
    return x3DFace4;
  else if (x3DPole0 == kNeighbor)
    return x3DPole2;
  else if (x3DPole1 == kNeighbor)
    return x3DPole3;
  else if (x3DPole2 == kNeighbor)
    return x3DPole0;
  else if (x3DPole3 == kNeighbor)
    return x3DPole1;
  else if (x3DPole4 == kNeighbor)
    return x3DPole6;
  else if (x3DPole5 == kNeighbor)
    return x3DPole7;
  else if (x3DPole6 == kNeighbor)
    return x3DPole4;
  else if (x3DPole7 == kNeighbor)
    return x3DPole5;
  else if (x3DPole8 == kNeighbor)
    return x3DPole10;
  else if (x3DPole9 == kNeighbor)
    return x3DPole11;
  else if (x3DPole10 == kNeighbor)
    return x3DPole8;
  else if (x3DPole11 == kNeighbor)
    return x3DPole9;
  else if (x3DCorner0 == kNeighbor)
    return x3DCorner7;
  else if (x3DCorner1 == kNeighbor)
    return x3DCorner6;
  else if (x3DCorner2 == kNeighbor)
    return x3DCorner5;
  else if (x3DCorner3 == kNeighbor)
    return x3DCorner4;
  else if (x3DCorner4 == kNeighbor)
    return x3DCorner3;
  else if (x3DCorner5 == kNeighbor)
    return x3DCorner2;
  else if (x3DCorner6 == kNeighbor)
    return x3DCorner1;
  else if (x3DCorner7 == kNeighbor)
    return x3DCorner0;
  else
    return x3DNeighborEnd;
}

/* TODO(den4gr)
   need to create a function that receives a particular neighbor segment,
   and then waits on all of the mpi_requests that were passed in from
   the Isends.
 */
void receiveNewGhostZones(const NeighborTag3D kNeighbor, Node* node,
        const int kBorder[3], const int kSegmentSize) {
  const int kNoNeighbor = -1;
  int my_rank = node->getRank();
  const int kNumTotalBlocks = node->numTotalSubDomains();
  for (unsigned int block_index = 0;
          block_index < kNumTotalBlocks;
          ++block_index) {
    SubDomain* data_block = node->globalGetSubDomain(block_index);
    int index_of_received_block = kNoNeighbor;
    /* received block may not have been for the previous block, due to
        the fact that 2 nodes may have many blocks that must communicate */
    const int kSourceRank =
            data_block->getNeighborLoc(static_cast<int> (kNeighbor));
    if (kSourceRank != my_rank && kSourceRank != kNoNeighbor)
      receiveSegment(node, &index_of_received_block);
  }
}

double benchmarkSubDomainCopy(SubDomain* sub_domain) {
  struct timeval start, stop;
  int size = 0;
  const int kNumberIterations = 20;
  double time_elapsed = 0.0;

  gettimeofday(&start, NULL);
  const int kDimensionality = sub_domain->getDimensionality();
  const int kLength[] = {sub_domain->getLength(0),
    sub_domain->getLength(1),
    sub_domain->getLength(2)};
  const int kTypeSize = sizeof (DTYPE);
  if (kDimensionality == 3)
    size = kLength[0] * kLength[1] * kLength[2];
  else if (kDimensionality == 2)
    size = kLength[0] * kLength[1];
  else if (kDimensionality == 1)
    size = kLength[0];

  scoped_array<DTYPE> destination(new DTYPE[size]());
  for (int iteration = 0; iteration < kNumberIterations; ++iteration) {
    memcpy(destination.get(), sub_domain->getBuffer(), size * kTypeSize);
    memcpy(sub_domain->getBuffer(), destination.get(), size * kTypeSize);
  }
  gettimeofday(&stop, NULL);
  time_elapsed = secondsElapsed(start, stop) / kNumberIterations;
  return 1 / time_elapsed;
}

void receiveNewGhostZones(const NeighborTag2D kNeighbor, Node* node,
        const int kBorder[3], const int kSegmentSize) {
  const int kNoNeighbor = -1;
  int my_rank = -1, result = 0;
  result = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  mpiCheckError(result);

  bool copy_to_mpi_buffer = false;
  const int kReceiveIndex = 1;
  for (unsigned int blockIndex = 0;
          blockIndex < node->numTotalSubDomains();
          ++blockIndex) {
    SubDomain* data_block = node->globalGetSubDomain(blockIndex);
    int indexOfIntendedBlock = -1;
    /* received block may not have been for the previous block, due to
        the fact that 2 nodes may have many blocks that must communicate */
    const int kSourceRank =
            data_block->getNeighborLoc(static_cast<int> (kNeighbor));

    if (kSourceRank != my_rank && kSourceRank != kNoNeighbor) {
      receiveSegment(node, &indexOfIntendedBlock);
    }
  }
}

int getNumberDimensions(const SubDomain* sub_domain) {
  int number_dimensions = 0;
  for (int i = 0; i < 3 && 0 < sub_domain->getLength(i); ++i)
    ++number_dimensions;
  return number_dimensions;
}

/**
    @brief how large is the largest segment in the data block, given the border
 * @param dataBlock the data block that contains the lengths we will use
 * @param kBorder the dimensions of the ghost zones in the data block
 * @return number of elements
 */
int getMaxSegmentSize(const SubDomain* dataBlock, const int kBorder[3],
        const int kNumberDimensions) {
  if (3 == kNumberDimensions)
    return getMaxSegmentSize3D(dataBlock, kBorder);
  else if (2 == kNumberDimensions)
    return getMaxSegmentSize2D(dataBlock, kBorder);
  else
    return 1;
}

int getMaxSegmentSize2D(const SubDomain* dataBlock, const int kBorder[3]) {
  const int kHeight = dataBlock->getLength(0) - 2 * kBorder[0];
  const int kWidth = dataBlock->getLength(1) - 2 * kBorder[1];
  int corner_size = 0, max_pole_size = 0;
  /* check size of 4 poles */
  max_pole_size = max(kBorder[0] * kWidth, kBorder[1] * kHeight);
  corner_size = kBorder[0] * kBorder[1];
  return max(max_pole_size, corner_size);
}

int getMaxSegmentSize3D(const SubDomain* dataBlock, const int kBorder[3]) {
  const int kDepth = dataBlock->getLength(0) - 2 * kBorder[0];
  const int kWidth = dataBlock->getLength(1) - 2 * kBorder[1];
  const int kHeight = dataBlock->getLength(2) - 2 * kBorder[2];
  int maxSegmentSize = 0, maxPoleSize = 0;
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
  return max(maxSegmentSize, kBorder[0] * kBorder[1] * kBorder[2]);
}

void delete3DBuffer(const int kDim1, const int kDim2,
        DTYPE*** buffer) {
  for (int i = 0; i < kDim1; ++i) {
    for (int j = 0; j < kDim2; ++j) {
      delete buffer[i][j];
    }
    delete buffer[i];
  }
  delete [] buffer;
  buffer = NULL;
}

DTYPE*** new3DBuffer(const int kDim1, const int kDim2, const int kDim3) {
  DTYPE*** buffer = new DTYPE**[kDim1];
  for (int i = 0; i < kDim1; ++i) {
    buffer[i] = new DTYPE*[kDim2];
    for (int j = 0; j < kDim2; ++j) {
      buffer[i][j] = new DTYPE[kDim3]();
    }
  }
  return buffer;
}

void exchangeGhostZones2D(Node* node, const int kBorder[3],
        MPI_Request* requests) {
  int numberMessagesSent = 0, segmentSize = 0;
  for (NeighborTag2D neighbor = x2DNeighborBegin;
          neighbor < x2DNeighborEnd;
          ++neighbor) {
    sendNewGhostZones(neighbor, node, kBorder, requests, &segmentSize,
            &numberMessagesSent);
  }
  for (NeighborTag2D neighbor = x2DNeighborBegin;
          neighbor < x2DNeighborEnd;
          ++neighbor) {
    receiveNewGhostZones(neighbor, node, kBorder, segmentSize);
  }
  int result = MPI_Waitall(numberMessagesSent, requests, MPI_STATUSES_IGNORE);
  mpiCheckError(result);
}

void sendGhostZones2D(Node* node, const int kBorder[3],
        int* num_messages_sent) {
  int segment_size = 0;
  const int kNumBlocks = node->numTotalSubDomains();
  if (kNumBlocks == 0) return;

  /* send and receive buffers for each block */
  if (NULL == mpi_requests) {
    const SubDomain* dataBlock = node->globalGetSubDomain(0);
    const int kNumberDimensions = getNumberDimensions(dataBlock);
    /* one non-blocking send per block */
    const int kTotalSegments = kNumNeighbors3D * kNumBlocks;
    mpi_requests = new MPI_Request[kTotalSegments];
    initAllGhostZoneTypes(dataBlock, kNumberDimensions, kBorder, mpi_types,
            mpi_sub_types, segment_offsets);
  }
  for (NeighborTag2D neighbor = x2DNeighborBegin;
          neighbor < x2DNeighborEnd;
          ++neighbor) {
    sendNewGhostZones(neighbor, node, kBorder, mpi_requests, &segment_size,
            num_messages_sent);
  }
}

void receiveGhostZones2D(Node* node, const int kBorder[3],
        const int kNumMessagesSent, MPI_Request* requests) {
  int segmentSize = 0;
  for (NeighborTag2D neighbor = x2DNeighborBegin;
          neighbor < x2DNeighborEnd;
          ++neighbor) {
    receiveNewGhostZones(neighbor, node, kBorder, segmentSize);
  }
  int result = MPI_Waitall(kNumMessagesSent, requests, MPI_STATUSES_IGNORE);
  mpiCheckError(result);
}

static double total_gz_time = 0.0;
static double receive_gz_time = 0.0;

void exchangeGhostZones3D(Node* node, const int kBorder[3],
        MPI_Request* requests) {
  const int kMyRank = node->getRank();
  struct timeval total_start, total_end, receive_start, receive_end;
  int number_messages_sent = 0;
  int segment_size = 0;
  gettimeofday(&total_start, NULL);
  for (NeighborTag3D neighbor = x3DNeighborBegin;
          neighbor < x3DNeighborEnd;
          ++neighbor) {
    sendNewGhostZones(neighbor, node, kBorder, requests, &segment_size,
            &number_messages_sent);
  }
  gettimeofday(&receive_start, NULL);

  for (NeighborTag3D neighbor = x3DNeighborBegin;
          neighbor < x3DNeighborEnd;
          ++neighbor) {
    receiveNewGhostZones(neighbor, node, kBorder, segment_size);
  }
  gettimeofday(&receive_end, NULL);

  int result = MPI_Waitall(number_messages_sent, requests, MPI_STATUSES_IGNORE);
  mpiCheckError(result);
  gettimeofday(&total_end, NULL);
  receive_gz_time += secondsElapsed(receive_start, receive_end);
  total_gz_time += secondsElapsed(total_start, total_end);
}

void receiveGhostZones3D(Node* node, const int kBorder[3],
        const int kNumMessagesSent, MPI_Request* requests) {
  struct timeval receive_start, receive_end, total_start, total_stop;
  int segment_size = 0;

  gettimeofday(&total_start, NULL);
  gettimeofday(&receive_start, NULL);
  for (NeighborTag3D neighbor = x3DNeighborBegin;
          neighbor < x3DNeighborEnd;
          ++neighbor) {
    receiveNewGhostZones(neighbor, node, kBorder, segment_size);
  }
  gettimeofday(&receive_end, NULL);

  int result = MPI_Waitall(kNumMessagesSent, requests, MPI_STATUSES_IGNORE);
  mpiCheckError(result);
  gettimeofday(&total_stop, NULL);
  receive_gz_time += secondsElapsed(receive_start, receive_end);
  total_gz_time += secondsElapsed(total_start, total_stop);
}

void sendGhostZones3D(Node* node, const int kBorder[3],
        int* num_messages_sent) {
  struct timeval send_start, send_stop;
  *num_messages_sent = 0;
  int segment_size = 0;
  const int kNumBlocks = node->numTotalSubDomains();
  if (kNumBlocks == 0) return;

  /* send and receive buffers for each block */
  if (NULL == mpi_requests) {
    const SubDomain* dataBlock = node->globalGetSubDomain(0);
    const int kNumberDimensions = getNumberDimensions(dataBlock);
    /* one non-blocking send per block */
    const int kTotalSegments = kNumNeighbors3D * kNumBlocks;
    mpi_requests = new MPI_Request[kTotalSegments];
    initAllGhostZoneTypes(dataBlock, kNumberDimensions, kBorder, mpi_types,
            mpi_sub_types, segment_offsets);
  }

  gettimeofday(&send_start, NULL);
  for (NeighborTag3D neighbor = x3DNeighborBegin;
          neighbor < x3DNeighborEnd;
          ++neighbor) {
    sendNewGhostZones(neighbor, node, kBorder, mpi_requests, &segment_size,
            num_messages_sent);
  }
  gettimeofday(&send_stop, NULL);
  total_gz_time += secondsElapsed(send_start, send_stop);
}

void updateStart(Node* node, const int kBorder[3], int* num_messages_sent) {
  const int kNumBlocks = node->numTotalSubDomains();
  if (kNumBlocks == 0) return;
  const SubDomain* dataBlock = node->globalGetSubDomain(0);
  const int kNumberDimensions = dataBlock->getDimensionality();

  /* send and receive buffers for each block */
  if (3 == kNumberDimensions)
    sendGhostZones3D(node, kBorder, num_messages_sent);
  else if (2 == kNumberDimensions)
    sendGhostZones2D(node, kBorder, num_messages_sent);
}


void updateFinish(Node* node, const int kBorder[3],
        const int kNumMessagesSent) {
  // TODO don't want to use barrier make locking more fine-grained
  // use barrier to fix problem where some nodes are much faster and comm
  //  is getting mixed up because some nodes are receiving messages from
  //  different iterations.
  MPI_Barrier(MPI_COMM_WORLD);
  
  const int kNumBlocks = node->numTotalSubDomains();
  if (kNumBlocks == 0) return;
  
  const SubDomain* dataBlock = node->globalGetSubDomain(0);
  const int kNumberDimensions = dataBlock->getDimensionality();

  if (3 == kNumberDimensions)
    receiveGhostZones3D(node, kBorder, kNumMessagesSent, mpi_requests);
  else if (2 == kNumberDimensions)
    receiveGhostZones2D(node, kBorder, kNumMessagesSent, mpi_requests);
}

/*
   TODO(den4gr)
 */
void updateAllStaleData(Node* node, const int kBorder[3]) {
  // TODO don't want to use barrier make locking more fine-grained
  // use barrier to fix problem where some nodes are much faster and comm
  //  is getting mixed up because some nodes are receiving messages from
  //  different iterations.
  MPI_Barrier(MPI_COMM_WORLD);

  const int kNumBlocks = node->numTotalSubDomains();
  if (kNumBlocks == 0) return;
  const SubDomain* dataBlock = node->globalGetSubDomain(0);
  const int kNumberDimensions = getNumberDimensions(dataBlock);

  /* send and receive buffers for each block */
  if (NULL == mpi_requests) {
    /*{
      int i = 0;
      char hostname[256];
      gethostname(hostname, sizeof (hostname));
      printf("PID %d on %s ready for attach\n", getpid(), hostname);
      fflush(stdout);
      while (0 == i)
        sleep(5);
    } // */
    /* one non-blocking send per block */
    const int kTotalSegments = kNumNeighbors3D * kNumBlocks;
    mpi_requests = new MPI_Request[kTotalSegments];
    initAllGhostZoneTypes(dataBlock, kNumberDimensions, kBorder, mpi_types,
            mpi_sub_types, segment_offsets);
  }
  if (3 == kNumberDimensions)
    exchangeGhostZones3D(node, kBorder, mpi_requests);
  else if (2 == kNumberDimensions)
    exchangeGhostZones2D(node, kBorder, mpi_requests);
}

void cleanupComm(const int kNumBlocks) {
  if (NULL != mpi_requests) {
    delete [] mpi_requests;
    mpi_requests = NULL;
  }
}
