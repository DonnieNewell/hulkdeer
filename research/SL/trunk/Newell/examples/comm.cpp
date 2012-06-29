#include "comm.h"
double secondsElapsed(struct timeval start, struct timeval stop) {
    return static_cast<double>((stop.tv_sec - start.tv_sec) +
                                (stop.tv_usec - start.tv_usec)/1000000.0);
}
void sendDataToNode(const int rank, int device, SubDomain* s) {
  // first send number of dimensions
  int numDim  = 0;
  MPI_Request reqs[8];
  int length[3];
  int offset[3];
  const int kNumNeighbors3D = 26;
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
  MPI_Isend(static_cast<void*>(s->getNeighbors()), kNumNeighbors3D, MPI_INT,
            rank, xNeighbors, MPI_COMM_WORLD,  &reqs[7]);

  // third send data
  // first we have to stage the data into contiguous memory
  int total_size = 1;
  for (int i = 0; i < numDim; ++i) {
    total_size *= length[i];
  }
  MPI_Isend(static_cast <void*>(s->getBuffer()), total_size, MPI_INT, rank,
             xData, MPI_COMM_WORLD, &reqs[4]);
  MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
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
  int total = n->numTotalSubDomains();
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

void benchmarkNode(Node* n, SubDomain* s) {
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
SubDomain* receiveDataFromNode(int rank, int* device) {
  MPI_Request reqs[7];
  int numDim =  0;
  int id[3]     = {-1, -1, -1};
  int gridDim[3]= {-1, -1, -1};
  int length[3];
  int offset[3];
  const int kNumNeighbors3D = 26;
  int neighbors[kNumNeighbors3D] = { 0 };

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
  MPI_Irecv(static_cast<void*>(neighbors), kNumNeighbors3D, MPI_INT, rank,
            xNeighbors, MPI_COMM_WORLD,  &reqs[6]);

  MPI_Waitall(7, reqs, MPI_STATUSES_IGNORE);

  SubDomain *s = new SubDomain(id, offset[0], length[0], offset[1],
                                   length[1], offset[2], length[2], gridDim[0],
                                   gridDim[1], gridDim[2], neighbors);
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
                          int* segmentOffset, SubDomain* dataBlock,
                          const int kBorder[3], const bool kBlockToBuffer) {
  if (xCorner0 == neighbor) {
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
  } else if (xCorner1 == neighbor) {
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
  } else if (xCorner2 == neighbor) {
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
  } else if (xCorner3 == neighbor) {
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
  } else if (xCorner4 == neighbor) {
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
  } else if (xCorner5 == neighbor) {
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
  } else if (xCorner6 == neighbor) {
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
  } else if (xCorner7 == neighbor) {
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

void getFaceDimensions(NeighborTag neighbor, int* segmentLength,
    int* segmentOffset, SubDomain* dataBlock,
    const int kBorder[3], const bool kBlockToBuffer) {
  if (xFace0 == neighbor) {
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
  } else if (xFace1 == neighbor) {
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
  } else if (xFace2 == neighbor) {
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
  } else if (xFace3 == neighbor) {
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
  } else if (xFace4 == neighbor) {
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
  } else if (xFace5 == neighbor) {
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

void getPoleDimensions(NeighborTag neighbor, int* segmentLength,
                          int* segmentOffset, SubDomain* dataBlock,
                          const int kBorder[3], const bool kBlockToBuffer) {
  if (xPole0 == neighbor) {
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
  } else if (xPole1 == neighbor) {
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
  } else if (xPole2 == neighbor) {
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
  } else if (xPole3 == neighbor) {
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
  } else if (xPole4 == neighbor) {
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
  } else if (xPole5 == neighbor) {
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
  } else if (xPole6 == neighbor) {
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
  } else if (xPole7 == neighbor) {
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
  } else if (xPole8 == neighbor) {
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
  } else if (xPole9 == neighbor) {
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
  } else if (xPole10 == neighbor) {
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
  } else if (xPole11 == neighbor) {
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

// TODO(den4gr)
void getSegmentDimensions(NeighborTag neighbor, int* segmentLength,
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

//TODO(den4gr)
/* copies to/from the buffer based on the bool flag */
void copySegment(NeighborTag neighbor, SubDomain* dataBlock,
    DTYPE* buffer, const int kBorder[3], const bool kBlockToBuffer,
    int* segmentSize) {
  int segmentLength[3] = { 0 };
  int segmentOffset[3] = { 0 };
  int blockLength[3] = {  dataBlock->getLength(0),
                          dataBlock->getLength(1),
                          dataBlock->getLength(2) };
  getSegmentDimensions(neighbor, segmentLength, segmentOffset, dataBlock,
                        kBorder, kBlockToBuffer);
  if (NULL != segmentSize) {
    *segmentSize =  segmentLength[0] *
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
        int blockIndex =  blockI * blockLength[1] * blockLength[2] +
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

/* Sends the ghost zone segment to the neighbor who needs it.
    This assumes that all blocks are sending the same neighbor,
    at the same time, with the same size. This allows us to just
    send the data, without any size or type information about the
    segment
*/
bool sendSegment(const NeighborTag kNeighbor, SubDomain* dataBlock,
                  DTYPE* sendBuffer, const int kSize, MPI_Request* request) {
  int sendRank = dataBlock->getNeighborLoc(kNeighbor);
  int blockLinearIndex = dataBlock->getNeighborIndex(kNeighbor);
  if (-1 < sendRank) {
    if (64 == sendRank) {
      printNeighbors(dataBlock);
    }
    MPI_Isend(static_cast<void*>(&blockLinearIndex), 1, MPI_INT, sendRank,
              xNeighborIndex, MPI_COMM_WORLD, &request[0]);
    MPI_Isend(static_cast<void*>(sendBuffer), kSize, MPI_INT, sendRank,
              xNeighborData, MPI_COMM_WORLD, &request[1]);
    return true;
  }
  return false;
}

bool receiveSegment(const NeighborTag kNeighbor, SubDomain* dataBlock,
                    DTYPE* receiveBuffer, const int kSegmentSize,
                    int* linearIndex) {
  int receiveRank = dataBlock->getNeighborLoc(kNeighbor);
  const int kNoNeighbor = -1;
  MPI_Status status;
  if (kNoNeighbor < receiveRank) {
    MPI_Recv(static_cast<void*>(linearIndex), 1, MPI_INT, receiveRank,
              xNeighborIndex, MPI_COMM_WORLD, &status);
    MPI_Recv(static_cast<void*>(receiveBuffer), kSegmentSize, MPI_INT,
              receiveRank, xNeighborData, MPI_COMM_WORLD, &status);
    return true;
  }
  return false;
}

/* TODO(den4gr)
    need to create a function that will send a particular neighbor segment
    for all blocks, and return the buffers and the MPI_Requests
*/
void sendNewGhostZones(const NeighborTag kNeighbor, Node* node,
                      const int kBorder[3], MPI_Request* requests,
                      DTYPE*** buffers, int* segmentSize,
                      int* numberMessagesSent) {
  const int kSendIndex = 0;
  for (unsigned int blockIndex = 0;
        blockIndex < node->numTotalSubDomains();
        ++blockIndex) {
    SubDomain* dataBlock = node->globalGetSubDomain(blockIndex);
    DTYPE* sendBuffer = buffers[blockIndex][kSendIndex];
    /* copy halo segment to buffer */
    bool copyBlockToBuffer = true;
    copySegment(kNeighbor, dataBlock, sendBuffer, kBorder,
                copyBlockToBuffer, segmentSize);
    bool didSend = sendSegment(kNeighbor, dataBlock, sendBuffer, *segmentSize,
                                &requests[*numberMessagesSent]);
    if (didSend)
      *numberMessagesSent += 2;
  }
}

/* This attrocious function is used in the exchange of ghost zones.
  It is used when you want to send a particular ghost zone segment
  to the neighboring cell. If you are sending Block A's face 0, to
  block B, then it will be stored in Block B's face 1 segment.
*/
NeighborTag getOppositeNeighbor3D(const NeighborTag kNeighbor) {
  if (xFace0 == kNeighbor)
    return xFace1;
  else if (xFace1 == kNeighbor)
    return xFace0;
  else if (xFace2 == kNeighbor)
    return xFace3;
  else if (xFace3 == kNeighbor)
    return xFace2;
  else if (xFace4 == kNeighbor)
    return xFace5;
  else if (xFace5 == kNeighbor)
    return xFace4;
  else if (xPole0 == kNeighbor)
    return xPole2;
  else if (xPole1 == kNeighbor)
    return xPole3;
  else if (xPole2 == kNeighbor)
    return xPole0;
  else if (xPole3 == kNeighbor)
    return xPole1;
  else if (xPole4 == kNeighbor)
    return xPole6;
  else if (xPole5 == kNeighbor)
    return xPole7;
  else if (xPole6 == kNeighbor)
    return xPole4;
  else if (xPole7 == kNeighbor)
    return xPole5;
  else if (xPole8 == kNeighbor)
    return xPole10;
  else if (xPole9 == kNeighbor)
    return xPole11;
  else if (xPole10 == kNeighbor)
    return xPole8;
  else if (xPole11 == kNeighbor)
    return xPole9;
  else if (xCorner0 == kNeighbor)
    return xCorner7;
  else if (xCorner1 == kNeighbor)
    return xCorner6;
  else if (xCorner2 == kNeighbor)
    return xCorner5;
  else if (xCorner3 == kNeighbor)
    return xCorner4;
  else if (xCorner4 == kNeighbor)
    return xCorner3;
  else if (xCorner5 == kNeighbor)
    return xCorner2;
  else if (xCorner6 == kNeighbor)
    return xCorner1;
  else if (xCorner7 == kNeighbor)
    return xCorner0;
  else
    return xNeighborEnd;
}

/* TODO(den4gr)
   need to create a function that receives a particular neighbor segment,
   and then waits on all of the mpi_requests that were passed in from
   the Isends.
 */
void receiveNewGhostZones(const NeighborTag kNeighbor,
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

int getMaxSegmentSize(SubDomain* dataBlock, const int kBorder[3]) {
  const int kDepth  = dataBlock->getLength(0) - 2 * kBorder[0];
  const int kWidth  = dataBlock->getLength(1) - 2 * kBorder[1];
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
  for(int i = 0; i < kDim1; ++i) {
    for(int j = 0; j < kDim2; ++j) {
      delete buffer[i][j];
    }
    delete buffer[i];
  }
  delete [] buffer;
  buffer = NULL;
}

DTYPE*** new3DBuffer(const int kDim1, const int kDim2, const int kDim3) {
  DTYPE*** buffer = new DTYPE**[kDim1];
  for(int i = 0; i < kDim1; ++i) {
    buffer[i] = new DTYPE*[kDim2];
    for(int j = 0; j < kDim2; ++j) {
      buffer[i][j] = new DTYPE[kDim3]();
    }
  }
  return buffer;
}

/*
   TODO(den4gr)
 */
void updateAllStaleData(Node* node, const int kBorder[3]) {
  const int kNumBlocks = node->numTotalSubDomains();
  const int kNumMessagesPerSegment = 2;
  const int kSendAndReceive = 2;
  const int kMaxSegmentSize = getMaxSegmentSize(node->getSubDomain(0), kBorder);

  /* one non-blocking send per block */
  MPI_Request* requests = new MPI_Request[kNumBlocks*kNumMessagesPerSegment];
  /* send and receive buffers for each block */
  DTYPE*** buffers = new3DBuffer(kNumBlocks, kSendAndReceive, kMaxSegmentSize);

  /* LOOP: over all halo segments */
  for ( NeighborTag neighbor = xNeighborBegin;
        neighbor < xNeighborEnd;
        ++neighbor) {
    int segmentSize = 0;
    int numberMessagesSent = 0;

    sendNewGhostZones(neighbor, node, kBorder, requests, buffers, &segmentSize,
                      &numberMessagesSent);
    NeighborTag oppositeNeighbor = getOppositeNeighbor3D(neighbor);
    receiveNewGhostZones(oppositeNeighbor, node, kBorder, buffers, segmentSize);
    MPI_Waitall(numberMessagesSent, requests, MPI_STATUSES_IGNORE);
  }
  delete [] requests;
  requests = NULL;
  delete3DBuffer(kNumBlocks, kSendAndReceive, kMaxSegmentSize, buffers);
  buffers = NULL;
}


