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
#include <fstream>
#include <sstream>
#include <string>
#include "../comm.h"
#include "../Cluster.h"
#include "../Decomposition.h"
#include "../Balancer.h"
#include "../Model.h"
#include "cell.h"
#include "ompCell.h"
#include "distributedCell.h"
#define PYRAMID_HEIGHT 1

const int kCPUIndex = -1;

void benchmarkMyself(Node* n, SubDomain* pS, int timesteps, int bornMin,
                      int bornMax, int dieMin, int dieMax) {
  // receive results for each device
  unsigned int total = n->getNumChildren()+1;
  MPI_Request req[2];
  double *weight = new double[total];
  double *edgeWeight = new double[total-1];
  SubDomain *s = NULL;
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
      processSubDomain(device-1, s, timesteps, bornMin, bornMax, dieMin,dieMax);
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


void receiveData(int rank, Node* n, bool processNow, int pyramidHeight,
                  int bornMin, int bornMax, int dieMin, int dieMax) {
  // receive number of task blocks that will be sent
  int numTaskBlocks = 0;
  MPI_Status stat;
  MPI_Recv(static_cast<void*>(&numTaskBlocks), 1, MPI_INT, rank, xNumBlocks,
            MPI_COMM_WORLD, &stat);
  struct timeval start, end;
  double receiveDataTime = 0.0, processBlockTime = 0.0;
  for (int block = 0; block < numTaskBlocks; ++block) {
    SubDomain* s = NULL;
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

void processSubDomain(int device, SubDomain *task, int timesteps,
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
                dieMax);
  } else {
    // run on GPU
    gettimeofday(&start, NULL);
    runCell(buff, depth, height, width, timesteps, bornMin, bornMax, dieMin,
            dieMax, device);
    gettimeofday(&end, NULL);
  }
}

double benchmarkPCIBus(SubDomain* pS, int gpuIndex) {
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
// TODO(den4gr)
/*
  takes a subdomain containing results and copies it into original
  buffer, accounting for invalid ghost zone around edges
*/
void copy_result_block(DTYPE* buffer, SubDomain* s, const int kBorder[3],
                        const int kBufferSize[3]) {
  const int kLength[3] = { s->getLength(0) - 2 * kBorder[0],
                          s->getLength(1) - 2 * kBorder[1],
                          s->getLength(2) - 2 * kBorder[2] };
  const int kOffset[3] = { s->getOffset(0) + kBorder[0],
                          s->getOffset(1) + kBorder[1],
                          s->getOffset(2) + kBorder[2] };
  for (int i = 0; i < kLength[0]; ++i) {
    for (int j = 0; j < kLength[1]; ++j) {
      for (int k = 0; k < kLength[2]; ++k) {
        int destI = i + kOffset[0];
        int destJ = j + kOffset[1];
        int destK = k + kOffset[2];
        int srcI = i + kBorder[0];
        int srcJ = j + kBorder[1];
        int srcK = k + kBorder[2];
        int destIndex = destI * kBufferSize[1] * kBufferSize[2] +
                        destJ * kBufferSize[2] +
                        destK;
        int srcIndex = srcI * kLength[1] * kLength[2] +
                        srcJ * kLength[2] +
                        srcK;
        buffer[destIndex] = s->getBuffer()[srcIndex];
      }
    }
  }
}

void copy_results(DTYPE* buffer, Cluster* cluster, const int kBorder[3],
                  const int kBufferSize[3]) {
  if (NULL == buffer) return;

  /* get work from all parents and children in cluster */
  for (unsigned int n = 0; n < cluster->getNumNodes(); ++n) {
    Node &node = cluster->getNode(n);
    unsigned int num = node.numSubDomains();
    for (unsigned int block =0; block < num; ++block) {
      copy_result_block(buffer, node.getSubDomain(block), kBorder, kBufferSize);
    }

    for (unsigned int c = 0; c < node.getNumChildren(); ++c) {
      Node* child = &(node.getChild(c));
      num = child->numSubDomains();

      for (unsigned int block =0; block < num; ++block) {
        copy_result_block(buffer, child->getSubDomain(block), kBorder,
                          kBufferSize);
      }
    }
  }
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
                  const int kStencilSize[3], const int kBornMin,
                  const int kBornMax, const int kDieMin, const int kDieMax) {
  int currentPyramidHeight = kPyramidHeight;
  const int kFirstIteration = 0;
  for (int iter = kFirstIteration; iter < kIterations; iter += kPyramidHeight) {
    if (iter + kPyramidHeight > kIterations) {
      currentPyramidHeight = kIterations - iter;
    }
    int staleBorder[3] = {  kStencilSize[0] * currentPyramidHeight,
                            kStencilSize[1] * currentPyramidHeight,
                            kStencilSize[2] * currentPyramidHeight };
    /* The data is initially sent with the ghost zones, but since
        we actually process each subdomain interleaved with the communication,
      in receiveData, we have to update the stale cells starting with the
      first iteration. Note, this is why the number of iterations passed
      into this function should be totalIterations - pyramidHeight, due to
      the previously mentioned reason. */
    if ((machine->getRank() == 0 && iter > 0) || (machine->getRank() > 0))
      updateAllStaleData(machine, staleBorder);
    processCPUWork(machine, currentPyramidHeight, kBornMin, kBornMax,
        kDieMin, kDieMax);
    processGPUWork(machine, currentPyramidHeight, kBornMin, kBornMax,
        kDieMin, kDieMax);
  }
}

void getResultsFromCluster(Cluster* cluster) {
  /* TODO(den4gr) receives results, needs to be asynchronous */
  const bool kNoInterleavedCompute = false;
  for (unsigned int nodeRank = 1;
          nodeRank < cluster->getNumNodes();
          ++nodeRank) {
    receiveData(nodeRank, &(cluster->getNode(nodeRank)), kNoInterleavedCompute,
                1, 0, 0, 0, 0);
  }
}

void sendWorkToCluster(Cluster* cluster) {
  /* TODO(den4gr) needs to be parallel.
      send the work to each node. */
  for (unsigned int node = 1; node < cluster->getNumNodes(); ++node) {
    sendData(&(cluster->getNode(node)));
  }
}

void benchmarkCluster(Cluster* cluster, SubDomain* data,
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
  std::stringstream ss;
  ss << rank;
  string filename = "cell.log.";
  filename += ss.str();
  std::ofstream log(filename.c_str(), std::ios_base::out);
  int new_stencil_size[3] = {stencil_size.z, stencil_size.y, stencil_size.x};
  int deviceCount = 0;
  const int kPyramidHeight = 1;
  const int kBorder[3] = { new_stencil_size[0] * kPyramidHeight,
                            new_stencil_size[1] * kPyramidHeight,
                            new_stencil_size[2] * kPyramidHeight };
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
    //printCluster(*cluster);  // DEBUG
    //printBlockLocations(*cluster);  // DEBUG
    balance_sec = secondsElapsed(balance_start, balance_end);
    fprintf(stderr, "***********\nBALANCE TIME: %f seconds.\n", balance_sec);
    gettimeofday(&process_start, NULL);
    log << "sending work to cluster.\n";
    sendWorkToCluster(cluster);
    // TODO(den4gr) Is this a deep copy??
    // root's work is in the first node
    myWork = cluster->getNode(0);
    /* PROCESS ROOT NODE WORK */
    gettimeofday(&comp_start, NULL);
    log << "processing root work.\n";
    processWork(&myWork, iterations, kPyramidHeight, new_stencil_size,
                bornMin, bornMax, dieMin, dieMax);
    gettimeofday(&comp_end, NULL);
    time_root_compute= secondsElapsed(comp_start, comp_end);
    fprintf(stdout, "*********\nroot processing time: %f sec\n",
        time_root_compute);

    log << "getting results from cluster.\n";
    gettimeofday(&rec_start, NULL);
    getResultsFromCluster(cluster);
    gettimeofday(&rec_end, NULL);

    copy_results(data, cluster, kBorder, numElements);

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
    log << "[" << rank <<"] sending number of children to root.\n";
    sendNumberOfChildren(0, deviceCount);
    benchmarkMyself(&myWork, NULL, iterations, bornMin, bornMax,
        dieMin, dieMax);
    receiveData(0, &myWork, kInterleaveProcessing, kPyramidHeight,
                bornMin, bornMax, dieMin, dieMax);
    remainingIterations = iterations - kPyramidHeight;
    processWork(&myWork, remainingIterations, kPyramidHeight, new_stencil_size,
                bornMin, bornMax, dieMin, dieMax);
    // send my work back to the root
    myWork.setRank(0);
    sendData(&myWork);
  }
  log.close();
}
