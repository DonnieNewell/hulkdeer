#include "comm.h"

double secondsElapsed(struct timeval start, struct timeval stop) {
  return static_cast<double> ((stop.tv_sec - start.tv_sec) +
          (stop.tv_usec - start.tv_usec) / 1000000.0);
}

void sendDataToNode(const int rank, int device, SubDomain* s) {
  // first send number of dimensions
  int numDim = 0;
  MPI_Request reqs[8];
  int length[3];
  int offset[3];
  const int kNumNeighbors3D = 26;
  const int kNumNeighbors2D = 8;
  const int kNumNeighbors1D = 2;
  int num_neighbors = 0;
  const int* kTmpId = s->getId();
  const int* kTmpGridDim = s->getGridDim();
  int tmpId[3] = {kTmpId[0], kTmpId[1], kTmpId[2]};
  int tmpGridDim[3] = {kTmpGridDim[0], kTmpGridDim[1], kTmpGridDim[2]};
  MPI_Isend(static_cast<void*> (tmpId), 3, MPI_INT, rank, xId,
          MPI_COMM_WORLD, &reqs[0]);
  MPI_Isend(static_cast<void*> (tmpGridDim), 3, MPI_INT, rank, xGridDim,
          MPI_COMM_WORLD, &reqs[1]);
  MPI_Isend(static_cast<void*> (&device), 1, MPI_INT, rank, xDevice,
          MPI_COMM_WORLD, &reqs[2]);
  for (int i = 0; i < 3; ++i) {
    length[i] = s->getLength(i);
    offset[i] = s->getOffset(i);

    if (length[i] > 0)
      ++numDim;
    //fprintf(stderr, "length[%d]:%d, numDim:%d\n", i, length[i], numDim);
  }
  //fprintf(stderr, "sending numDim:%d to rank:%d\n", numDim, rank);
  //fprintf(stderr, "len = {%d, %d, %d}.\n", length[0], length[1], length[2]);
  MPI_Isend(static_cast<void*> (&numDim), 1, MPI_INT, rank, xDim,
          MPI_COMM_WORLD, &reqs[3]);
  MPI_Isend(static_cast<void*> (length), 3, MPI_INT, rank, xLength,
          MPI_COMM_WORLD, &reqs[4]);
  MPI_Isend(static_cast<void*> (offset), 3, MPI_INT, rank, xOffset,
          MPI_COMM_WORLD, &reqs[5]);
  if (3 == numDim)
    num_neighbors = kNumNeighbors3D;
  else if (2 == numDim)
    num_neighbors = kNumNeighbors2D;
  else if (1 == numDim)
    num_neighbors = kNumNeighbors1D;
  MPI_Isend(static_cast<void*> (s->getNeighbors()), num_neighbors, MPI_INT,
          rank, xNeighbors, MPI_COMM_WORLD, &reqs[6]);

  // third send data
  // first we have to stage the data into contiguous memory
  int total_size = 1;
  for (int i = 0; i < numDim; ++i) {
    total_size *= length[i];
  }
  MPI_Isend(static_cast<void*> (s->getBuffer()), total_size, SL_MPI_TYPE, rank,
          xData, MPI_COMM_WORLD, &reqs[7]);
  MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
}

void receiveNumberOfChildren(int numTasks, Cluster* cluster) {
  MPI_Request *reqs = new MPI_Request[numTasks - 1];

  int* numChildren = new int[numTasks - 1];

  for (int i = 0; i < numTasks - 1; i++) {
    // receive next count
    MPI_Irecv(static_cast<void*> (numChildren + i), 1, MPI_INT, i + 1, xChildren,
            MPI_COMM_WORLD, &(reqs[i]));
  }
  MPI_Waitall(numTasks - 1, reqs, MPI_STATUSES_IGNORE);
  for (int task = 0; task < numTasks - 1; task++) {
    cluster->getNode(task + 1).setNumChildren(numChildren[task]);
  }

  delete [] reqs;
  reqs = NULL;
  delete [] numChildren;
  numChildren = NULL;
}

void sendData(Node* n) {
  // count how many task blocks, total, are going to be sent
  int total = n->numTotalSubDomains();
  // send node number of blocks
  MPI_Request req;
  MPI_Isend(static_cast<void*> (&total), 1, MPI_INT, n->getRank(), xNumBlocks,
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

void benchmarkNode(Node* n, SubDomain* s) {
  struct timeval start, end;
  double total_sec = 0.0;
  gettimeofday(&start, NULL);
  // send task block to every device on that node
  sendDataToNode(n->getRank(), -1, s);
  fprintf(stderr, "sent data to Node %d.\n", n->getRank());
  gettimeofday(&end, NULL);

  total_sec = secondsElapsed(start, end);
  // how fast is the connection between root and child nodes
  // multiply by 2 to account for there and back
  n->setEdgeWeight(1 / (2 * total_sec));
  // receive results for each device
  unsigned int total = n->getNumChildren() + 1;

  MPI_Request req[2];
  double *task_per_sec = new double[total];
  double *edge_weight = new double[total - 1];

  // fprintf(stderr, "preparing to receive the benchmark data from %d.\n",n->getRank());
  MPI_Irecv(static_cast<void*> (task_per_sec), total, MPI_DOUBLE, n->getRank(),
          xWeight, MPI_COMM_WORLD, &req[0]);
  MPI_Irecv(static_cast<void*> (edge_weight), total - 1, MPI_DOUBLE, n->getRank(),
          xEdgeWeight, MPI_COMM_WORLD, &req[1]);
  MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
  fprintf(stderr, "received the benchmark data from %d.\n", n->getRank());
  // set the appropriate fields in the node and its children
  for (unsigned int device = 0; device < total; ++device) {
    double weight = task_per_sec[device];
    if (device == 0) {
      // the first weight is for the cpu
      fprintf(stderr, "setting node[%d] weight to %f.\n", n->getRank(), weight);
      n->setWeight(weight);
    } else {
      double edgeWeight = edge_weight[device - 1];
      fprintf(stderr, "setting node[%d].child[%d] weight to %f.\n",
              n->getRank(), device - 1, weight);
      n->getChild(device - 1).setWeight(weight);
      n->getChild(device - 1).setEdgeWeight(edgeWeight);
    }
  }
  // clean up
  delete [] task_per_sec;
  task_per_sec = NULL;
  delete [] edge_weight;
  edge_weight = NULL;
}

/* output variables: buf, size */
SubDomain* receiveDataFromNode(int rank, int* device) {
  MPI_Request reqs[7];
  int numDim = 0;
  int id[3] = {-1, -1, -1};
  int gridDim[3] = {-1, -1, -1};
  int length[3];
  int offset[3];
  int size = 0;
  const int kNumNeighbors3D = 26;
  int neighbors[kNumNeighbors3D] = {0};
  SubDomain *sub_domain = NULL;
  MPI_Status status;

  // receive dimensionality of data
  MPI_Irecv(static_cast<void*> (id), 3, MPI_INT, rank, xId, MPI_COMM_WORLD,
          &reqs[0]);
  MPI_Irecv(static_cast<void*> (gridDim), 3, MPI_INT, rank, xGridDim,
          MPI_COMM_WORLD, &reqs[1]);
  MPI_Irecv(static_cast<void*> (device), 1, MPI_INT, rank, xDevice,
          MPI_COMM_WORLD, &reqs[2]);
  MPI_Irecv(static_cast<void*> (&numDim), 1, MPI_INT, rank, xDim,
          MPI_COMM_WORLD, &reqs[3]);

  // receive size of data
  MPI_Irecv(static_cast<void*> (length), 3, MPI_INT, rank, xLength,
          MPI_COMM_WORLD, &reqs[4]);
  MPI_Irecv(static_cast<void*> (offset), 3, MPI_INT, rank, xOffset,
          MPI_COMM_WORLD, &reqs[5]);
  MPI_Irecv(static_cast<void*> (neighbors), kNumNeighbors3D, MPI_INT, rank,
          xNeighbors, MPI_COMM_WORLD, &reqs[6]);

  MPI_Waitall(7, reqs, MPI_STATUSES_IGNORE);
  if (2 == numDim) {
    sub_domain = new SubDomain(id, offset[0], length[0], offset[1],
            length[1], gridDim[0],
            gridDim[1], neighbors);
  } else if (3 == numDim) {
    sub_domain = new SubDomain(id, offset[0], length[0], offset[1],
            length[1], offset[2], length[2], gridDim[0],
            gridDim[1], gridDim[2], neighbors);
  }

  size = length[0];
  for (int i = 1; i < numDim; ++i) size *= length[i];

  MPI_Recv(static_cast<void*> (sub_domain->getBuffer()), size, SL_MPI_TYPE,
          rank, xData, MPI_COMM_WORLD, &status);

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
      segmentOffset[0] = kHeight - 2 * kBorder[0];
      segmentOffset[1] = kBorder[1];
    } else {
      segmentOffset[0] = kHeight - kBorder[0];
      segmentOffset[1] = 0;
    }
  } else if (x2DCorner1 == neighbor) {
    if (kBlockToBuffer) {
      segmentOffset[0] = kHeight - 2 * kBorder[0];
      segmentOffset[1] = kWidth - 2 * kBorder[1];
    } else {
      segmentOffset[0] = kHeight - kBorder[0];
      segmentOffset[1] = kWidth - kBorder[1];
    }
  } else if (x2DCorner2 == neighbor) {
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kWidth - 2 * kBorder[1];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = kWidth - kBorder[1];
    }
  } else if (x2DCorner3 == neighbor) {
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = 0;
    }
  }
}

void getFaceDimensions(NeighborTag3D neighbor, int* segmentLength,
        int* segmentOffset, SubDomain* dataBlock,
        const int kBorder[3], const bool kBlockToBuffer) {
  if (x3DFace0 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = dataBlock->getLength(1) - 2 * kBorder[1];
    segmentLength[2] = dataBlock->getLength(2) - 2 * kBorder[2];
    if (kBlockToBuffer) {
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
    if (kBlockToBuffer) {
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
    if (kBlockToBuffer) {
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
    if (kBlockToBuffer) {
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
    if (kBlockToBuffer) {
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
    if (kBlockToBuffer) {
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
        const int kBorder[3], const bool kBlockToBuffer) {
  const int kHeight = dataBlock->getLength(0);
  const int kWidth = dataBlock->getLength(1);
  if (x2DPole0 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kWidth - 2 * kBorder[1];
    if (kBlockToBuffer) {
      segmentOffset[0] = kHeight - 2 * kBorder[0];
      segmentOffset[1] = kBorder[1];
    } else {
      segmentOffset[0] = kHeight - kBorder[0];
      segmentOffset[1] = kBorder[1];
    }
  } else if (x2DPole1 == neighbor) {
    segmentLength[0] = kHeight - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kWidth - 2 * kBorder[1];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kWidth - kBorder[1];
    }
  } else if (x2DPole2 == neighbor) {
    segmentLength[0] = kBorder[0];
    segmentLength[1] = kWidth - 2 * kBorder[1];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
    } else {
      segmentOffset[0] = 0;
      segmentOffset[1] = kBorder[1];
    }
  } else if (x2DPole3 == neighbor) {
    segmentLength[0] = kHeight - 2 * kBorder[0];
    segmentLength[1] = kBorder[1];
    if (kBlockToBuffer) {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = kBorder[1];
    } else {
      segmentOffset[0] = kBorder[0];
      segmentOffset[1] = 0;
    }
  }
}

int getMPITagForSegment(NeighborTag2D segment) {
  return static_cast<MPITagType> (static_cast<int> (segment) + 20);
}

int getMPITagForSegment(NeighborTag3D segment) {
  return static_cast<MPITagType> (static_cast<int> (segment) + 30);
}

int getMPITagForSegmentData(NeighborTag2D segment) {
  return static_cast<MPITagType> (static_cast<int> (segment) + 200);
}

int getMPITagForSegmentData(NeighborTag3D segment) {
  return static_cast<MPITagType> (static_cast<int> (segment) + 300);
}
// TODO(den4gr)

void getSegmentDimensions(NeighborTag3D neighbor, int* segmentLength,
        int* segmentOffset, SubDomain* dataBlock,
        const int kBorder[3], const bool kBlockToBuffer) {
  if (isSegmentFace(neighbor)) {
    getFaceDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
            kBorder, kBlockToBuffer);
  } else if (isSegmentPole(neighbor)) {
    getPoleDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
            kBorder, kBlockToBuffer);
  } else if (isSegmentCorner(neighbor)) {
    getCornerDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
            kBorder, kBlockToBuffer);
  }
}

void getSegmentDimensions(NeighborTag2D neighbor, int* segmentLength,
        int* segmentOffset, SubDomain* dataBlock,
        const int kBorder[3], const bool kBlockToBuffer) {
  if (isSegmentPole(neighbor)) {
    getPoleDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
            kBorder, kBlockToBuffer);
  } else if (isSegmentCorner(neighbor)) {
    getCornerDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
            kBorder, kBlockToBuffer);
  }
}

//TODO(den4gr)

/* copies to/from the buffer based on the bool flag */
void copySegment(NeighborTag3D neighbor, SubDomain* dataBlock,
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
bool sendSegment(const NeighborTag3D kNeighbor, SubDomain* dataBlock,
        int* destination_block_index, DTYPE* sendBuffer, const int kSize,
        MPI_Request* request) {
  int sendRank = dataBlock->getNeighborLoc(kNeighbor);
  *destination_block_index = dataBlock->getNeighborIndex(kNeighbor);
  if (-1 < sendRank) {
    if (64 == sendRank) {
      printNeighbors(dataBlock);
    }
    const NeighborTag3D kOppositeNeighbor = getOppositeNeighbor3D(kNeighbor);
    int index_tag = getMPITagForSegment(kOppositeNeighbor);
    int data_tag = getMPITagForSegmentData(kOppositeNeighbor);

    MPI_Isend(static_cast<void*> (destination_block_index), 1, MPI_INT, sendRank,
            index_tag, MPI_COMM_WORLD, &request[0]);
    MPI_Isend(static_cast<void*> (sendBuffer), kSize, MPI_INT, sendRank,
            data_tag, MPI_COMM_WORLD, &request[1]);
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
bool sendSegment(const NeighborTag2D kNeighbor, SubDomain* dataBlock,
        int* destination_block_index, DTYPE* sendBuffer, const int kSize,
        MPI_Request* request) {
  int destination_rank = dataBlock->getNeighborLoc(static_cast<int> (kNeighbor));
  *destination_block_index = dataBlock->getNeighborIndex(kNeighbor);
  if (-1 < destination_rank) {
    const NeighborTag2D kOppositeNeighbor = getOppositeNeighbor2D(kNeighbor);
    int index_tag = getMPITagForSegment(kOppositeNeighbor);
    int data_tag = getMPITagForSegmentData(kOppositeNeighbor);

    //printf("sending %s to block:%d on rank:%d with index_tag:%d data_tag:%d\n",
     //       neighborString(kOppositeNeighbor), *destination_block_index,
       //     destination_rank, index_tag, data_tag);
    MPI_Isend(static_cast<void*> (destination_block_index), 1, MPI_INT,
            destination_rank, index_tag, MPI_COMM_WORLD, &request[0]);
    MPI_Isend(static_cast<void*> (sendBuffer), kSize, SL_MPI_TYPE,
            destination_rank, data_tag, MPI_COMM_WORLD, &request[1]);
    return true;
  }
  return false;
}

bool receiveSegment(const NeighborTag3D kNeighbor, SubDomain* dataBlock,
        DTYPE* receiveBuffer, const int kSegmentSize,
        int* linearIndex) {
  int source_rank = dataBlock->getNeighborLoc(kNeighbor);
  const int kNoNeighbor = -1;
  MPI_Status status;
  int error = -1;
  int index_tag = getMPITagForSegment(kNeighbor);
  int data_tag = getMPITagForSegmentData(kNeighbor);
  if (kNoNeighbor < source_rank) {
    error = MPI_Recv(static_cast<void*> (linearIndex), 1, MPI_INT, source_rank,
            index_tag, MPI_COMM_WORLD, &status);
    error |= MPI_Recv(static_cast<void*> (receiveBuffer), kSegmentSize, MPI_INT,
            source_rank, data_tag, MPI_COMM_WORLD, &status);
    if (MPI_SUCCESS != error)
      printf("ERROR: receiveSegment(): MPI_Recv().\n");
    return true;
  }
  return false;
}

bool receiveSegment(const NeighborTag2D kNeighbor, SubDomain* dataBlock,
        DTYPE* receiveBuffer, const int kSegmentSize,
        int* linearIndex) {

  int receiveRank = dataBlock->getNeighborLoc(static_cast<int> (kNeighbor));
  const int kNoNeighbor = -1;
  int index_tag = getMPITagForSegment(kNeighbor);
  int data_tag = getMPITagForSegmentData(kNeighbor);
  MPI_Status status;
  if (kNoNeighbor < receiveRank) {
    //printf("linear index before receive:%d \n", *linearIndex);
    //printf("about to receive segment %s from rank %d index_tag:%d data_tag:%d\n",
      //      neighborString(kNeighbor), receiveRank, index_tag, data_tag);
    MPI_Recv(static_cast<void*> (linearIndex), 1, MPI_INT, receiveRank,
            index_tag, MPI_COMM_WORLD, &status);
    if (MPI_SUCCESS != status.MPI_ERROR) printf("receiveSegment: ERROR in MPI_Recv for linearIndex.\n");
    //printf("received segment %s from rank %d for block:%d \n",
      //      neighborString(kNeighbor), status.MPI_SOURCE, *linearIndex);
    MPI_Recv(static_cast<void*> (receiveBuffer), kSegmentSize, SL_MPI_TYPE,
            receiveRank, data_tag, MPI_COMM_WORLD, &status);
    if (MPI_SUCCESS != status.MPI_ERROR) printf("receiveSegment: ERROR in MPI_Recv for receiveBuffer.\n");
    return true;
  }
  return false;
}

/* TODO(den4gr)
    need to create a function that will send a particular neighbor segment
    for all blocks, and return the buffers and the MPI_Requests
 */
void sendNewGhostZones(const NeighborTag3D kNeighbor, Node* node,
        const int kBorder[3], MPI_Request* requests, int* block_indices,
        DTYPE*** buffers, int* segmentSize,
        int* numberMessagesSent) {
  const int kSendIndex = 0;
  for (unsigned int blockIndex = 0;
          blockIndex < node->numTotalSubDomains();
          ++blockIndex) {
    SubDomain* dataBlock = node->globalGetSubDomain(blockIndex);
    DTYPE* sendBuffer = buffers[blockIndex][kSendIndex];
    int* destination_block = &block_indices[blockIndex];
    /* copy halo segment to buffer */
    bool copyBlockToBuffer = true;
    copySegment(kNeighbor, dataBlock, sendBuffer, kBorder,
            copyBlockToBuffer, segmentSize);
    bool didSend = sendSegment(kNeighbor, dataBlock, destination_block,
            sendBuffer, *segmentSize, &requests[*numberMessagesSent]);
    if (didSend)
      *numberMessagesSent += 2;
  }
}

void sendNewGhostZones(const NeighborTag2D kNeighbor, Node* node,
        const int kBorder[3], MPI_Request* requests, int* block_indices,
        DTYPE*** buffers, int* segmentSize,
        int* numberMessagesSent) {
  const int kSendIndex = 0;
  for (unsigned int blockIndex = 0;
          blockIndex < node->numTotalSubDomains();
          ++blockIndex) {
    SubDomain* dataBlock = node->globalGetSubDomain(blockIndex);
    DTYPE* sendBuffer = buffers[blockIndex][kSendIndex];
    int* destination_block = &block_indices[blockIndex];
    /* copy halo segment to buffer */
    bool copyBlockToBuffer = true;
    // printf("copySegment(%s, block:%d, buffer:%p, border{%d, %d}, true, segmentSize:%d)\n",
    //        neighborString(kNeighbor), dataBlock->getLinIndex(), sendBuffer,
    //        kBorder[0], kBorder[1], *segmentSize);
    copySegment(kNeighbor, dataBlock, sendBuffer, kBorder,
            copyBlockToBuffer, segmentSize);
    bool didSend = sendSegment(kNeighbor, dataBlock, destination_block,
            sendBuffer, *segmentSize, &requests[*numberMessagesSent]);
    if (didSend)
      *numberMessagesSent += 2;
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
void receiveNewGhostZones(const NeighborTag3D kNeighbor,
        Node* node,
        const int kBorder[3],
        DTYPE*** buffers,
        const int kSegmentSize) {
  bool copyBlockToBuffer = false;
  const int kReceiveIndex = 1;
  for (unsigned int blockIndex = 0;
          blockIndex < node->numTotalSubDomains();
          ++blockIndex) {
    SubDomain* dataBlock = node->globalGetSubDomain(blockIndex);
    DTYPE* receiveBuffer = buffers[blockIndex][kReceiveIndex];
    int indexOfIntendedBlock = -1;
    /* received block may not have been for the previous block, due to
        the fact that 2 nodes may have many blocks that must communicate */
    receiveSegment(kNeighbor, dataBlock, receiveBuffer, kSegmentSize,
            &indexOfIntendedBlock);
    if (-1 == indexOfIntendedBlock) continue;

    SubDomain* receivedBlock = node->getSubDomainLinear(indexOfIntendedBlock);
    copySegment(kNeighbor, receivedBlock, receiveBuffer, kBorder,
            copyBlockToBuffer, NULL);
  }
}

void receiveNewGhostZones(const NeighborTag2D kNeighbor,
        Node* node,
        const int kBorder[3],
        DTYPE*** buffers,
        const int kSegmentSize) {
  bool copyBlockToBuffer = false;
  const int kReceiveIndex = 1;
  for (unsigned int blockIndex = 0;
          blockIndex < node->numTotalSubDomains();
          ++blockIndex) {
    SubDomain* dataBlock = node->globalGetSubDomain(blockIndex);
    DTYPE* receiveBuffer = buffers[blockIndex][kReceiveIndex];
    int indexOfIntendedBlock = -1;
    /* received block may not have been for the previous block, due to
        the fact that 2 nodes may have many blocks that must communicate */
    receiveSegment(kNeighbor, dataBlock, receiveBuffer, kSegmentSize,
            &indexOfIntendedBlock);
    if (-1 == indexOfIntendedBlock) continue;

    SubDomain* receivedBlock = node->getSubDomainLinear(indexOfIntendedBlock);
    if (NULL == receivedBlock)
      printf("***COULDN'T FIND BLOCK %d***\n", indexOfIntendedBlock);
    copySegment(kNeighbor, receivedBlock, receiveBuffer, kBorder,
            copyBlockToBuffer, NULL);
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
        const int kDim3, DTYPE*** buffer) {
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

void exchangeGhostZones2D(Node* node, const int kBorder[3], int* block_indices,
        DTYPE*** buffers, MPI_Request* requests) {
  /* {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof (hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }  // */
  for (NeighborTag2D neighbor = x2DNeighborBegin;
          neighbor < x2DNeighborEnd;
          ++neighbor) {
    int segmentSize = 0;
    int numberMessagesSent = 0;

    sendNewGhostZones(neighbor, node, kBorder, requests, block_indices, buffers,
            &segmentSize, &numberMessagesSent);
    NeighborTag2D oppositeNeighbor = getOppositeNeighbor2D(neighbor);
    receiveNewGhostZones(oppositeNeighbor, node, kBorder, buffers, segmentSize);
    MPI_Waitall(numberMessagesSent, requests, MPI_STATUSES_IGNORE);
  }
}

void exchangeGhostZones3D(Node* node, const int kBorder[3], int* block_indices,
        DTYPE*** buffers, MPI_Request* requests) {
  for (NeighborTag3D neighbor = x3DNeighborBegin;
          neighbor < x3DNeighborEnd;
          ++neighbor) {
    int segmentSize = 0;
    int numberMessagesSent = 0;

    sendNewGhostZones(neighbor, node, kBorder, requests, block_indices, buffers,
            &segmentSize, &numberMessagesSent);
    NeighborTag3D oppositeNeighbor = getOppositeNeighbor3D(neighbor);
    receiveNewGhostZones(oppositeNeighbor, node, kBorder, buffers, segmentSize);
    MPI_Waitall(numberMessagesSent, requests, MPI_STATUSES_IGNORE);
  }
}

/*
   TODO(den4gr)
 */
void updateAllStaleData(Node* node, const int kBorder[3]) {
  const int kNumBlocks = node->numTotalSubDomains();
  const int kNumMessagesPerSegment = 2;
  const int kSendAndReceive = 2;
  const SubDomain* dataBlock = node->getSubDomain(0);
  const int kNumberDimensions = getNumberDimensions(dataBlock);
  const int kMaxSegmentSize = getMaxSegmentSize(dataBlock, kBorder,
          kNumberDimensions);

  /* one non-blocking send per block */
  MPI_Request* requests = new MPI_Request[kNumBlocks * kNumMessagesPerSegment];
  // printf("allocated requests array for node %d with %d elements.\n",
  //        node->getRank(), kNumBlocks * kNumMessagesPerSegment);

  /* send and receive buffers for each block */
  int* block_indices = new int[kNumBlocks]();
  DTYPE*** buffers = new3DBuffer(kNumBlocks, kSendAndReceive, kMaxSegmentSize);
  // printf("allocated send/receive array for node %d dimensions[%d][%d][%d].\n",
  //        node->getRank(), kNumBlocks, kSendAndReceive, kMaxSegmentSize);

  /* LOOP: over all halo segments */
  if (3 == kNumberDimensions)
    exchangeGhostZones3D(node, kBorder, block_indices, buffers, requests);
  else if (2 == kNumberDimensions)
    exchangeGhostZones2D(node, kBorder, block_indices, buffers, requests);

  delete [] block_indices;
  block_indices = NULL;
  delete [] requests;
  requests = NULL;
  delete3DBuffer(kNumBlocks, kSendAndReceive, kMaxSegmentSize, buffers);
  buffers = NULL;
}
