/*
Copyright 2012 Donald Newell
 */
#include <stdlib.h>
#include <stdio.h>
#ifndef WIN32
#include <sys/time.h>
#else
#include <time.h>
#endif
#include <limits>
#include <fstream>
#include <sstream>
#include "../comm.h"
#include "./hotspot.h"
#include "./ompHotspot.h"
#include "../Cluster.h"
#include "../Decomposition.h"
#include "../Balancer.h"
#include "../Model.h"
#include "./distributedHotspot.h"
#define PYRAMID_HEIGHT 1

const int kCPUIndex = -1;
static DTYPE* global_data = NULL;

void benchmarkMyself(Node* node, SubDomain* sub_domain, int iterations,
        const int kPyramidHeight, float step_div_Cap, float Rx, float Ry,
        float Rz) {
  // fprintf(stderr, "benchmarkMyself(n:%p, pS:%p, timesteps:%d.\n", n, pS, timesteps);
  // receive results for each device
  int total_devices = static_cast<int> (node->getNumChildren()) + 1;
  double *compute_rate = new double[2 * total_devices - 1]();
  double *comm_rate = &compute_rate[total_devices];
  const int kRootRank = 0;
  SubDomain *benchmark_block = NULL;
  int intended_device = -2;

  if (sub_domain == NULL) {
    benchmark_block = receiveDataFromNode(kRootRank, &intended_device);
    if (-1 != intended_device) {
      fprintf(stderr, "data should be sent to device: -1, not:%d\n", intended_device);
    }
  } else {
    benchmark_block = sub_domain;
  }
  for (int device_index = 0; device_index < total_devices; ++device_index) {
    struct timeval start, end;
    double total_sec = 0.0;

    gettimeofday(&start, NULL);
    int current_pyramid_height = kPyramidHeight;
    for (int itr = 0; itr < iterations; itr += current_pyramid_height) {
      if (itr + current_pyramid_height > iterations)
        current_pyramid_height = iterations - itr;
      processSubDomain(device_index - 1, benchmark_block,
              current_pyramid_height, step_div_Cap, Rx, Ry, Rz);
    }
    gettimeofday(&end, NULL);
    total_sec = secondsElapsed(start, end);
    compute_rate[device_index] = iterations / total_sec;
    fprintf(stderr, "[%d]device:%d of %d processes %f iter/sec.\n",
            node->getRank(), device_index - 1, total_devices, compute_rate[device_index]);
    if (device_index == 0) {
      node->setWeight(compute_rate[device_index]);
      node->setEdgeWeight(benchmarkSubDomainCopy(benchmark_block));
    } else {
      node->getChild(device_index - 1).setWeight(compute_rate[device_index]);
      comm_rate[device_index - 1] = benchmarkPCIBus(benchmark_block, device_index - 1);
      node->getChild(device_index - 1).setEdgeWeight(comm_rate[device_index - 1]);
    }
  }

  if (NULL == sub_domain) {
    // send the result back to the host
    MPI_Send(static_cast<void*> (compute_rate), 2 * total_devices - 1,
            MPI_DOUBLE, kRootRank, xWeight, MPI_COMM_WORLD);
  }

  // clean up
  delete [] compute_rate;
  compute_rate = comm_rate = NULL;
  if (sub_domain == NULL) {
    delete benchmark_block;
    benchmark_block = NULL;
  }
}

void runDistributedHotspotSetData(DTYPE *data, int num_elements) {
  // TODO (donnie) this needs to have the MPI data type set dynamically
  const int kRootIndex = 0;
  global_data = data;
  // printf("MPI_Bcast()\n");
  MPI_Bcast(global_data, num_elements, SL_MPI_TYPE, kRootIndex, MPI_COMM_WORLD);
  runHotspotSetData(global_data, num_elements);
  // printf("runHotspotSetData()\n");
  runOMPHotspotSetData(global_data, num_elements);
  // printf("runOMPHotspotSetData()\n");
}

void receiveData(int rank, Node* n, bool processNow, int pyramidHeight,
        float step_div_Cap, float Rx, float Ry, float Rz) {
  // receive number of task blocks that will be sent
  int numTaskBlocks = 0;
  MPI_Status stat;
  MPI_Recv(static_cast<void*> (&numTaskBlocks), 1, MPI_INT, rank, xNumBlocks,
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
        processSubDomain(device, s, pyramidHeight, step_div_Cap,
                Rx, Ry, Rz);
        gettimeofday(&end, NULL);
        processBlockTime += secondsElapsed(start, end);
      }
      // add block to cpu queue
      n->addSubDomain(s);
    } else {
      if (processNow) {
        gettimeofday(&start, NULL);
        processSubDomain(device, s, pyramidHeight, step_div_Cap, Rx, Ry,
                Rz);
        gettimeofday(&end, NULL);
        processBlockTime += secondsElapsed(start, end);
      }
      // add block to gpu queue
      n->getChild(device).addSubDomain(s);
    }
  }
  fprintf(stderr, "[%d] comm. time %f, process time %f.\n",
          n->getRank(), receiveDataTime, processBlockTime);
  //runHotspotCleanup();
}

void getNumberOfChildren(int* numChildren) {
  /* check to see how many NVIDIA GPU'S ARE AVAILABLE */
  cudaError_t err = cudaGetDeviceCount(numChildren);
  if (cudaSuccess == cudaErrorNoDevice) {
    *numChildren = 0;
  } else if (cudaSuccess != err) {
    fprintf(stderr, "no CUDA-enabled devices.\n");
    *numChildren = 0;
  }
}

void sendNumberOfChildren(const int dest_rank, const int kNumChildren) {
  MPI_Request req;
  int number_children = kNumChildren;
  MPI_Isend(static_cast<void*> (&number_children), 1, MPI_INT, dest_rank,
          xChildren, MPI_COMM_WORLD, &req);
  MPI_Waitall(1, &req, MPI_STATUSES_IGNORE);
}

void processSubDomain(int device, SubDomain *task, int kPyramidHeight,
        float step_div_Cap, float Rx, float Ry, float Rz) {
  // DTYPE?
  DTYPE* buff = task->getBuffer();
  int height = task->getLength(0);
  int width = task->getLength(1);
  struct timeval start, end;
  if (-1 == device) {
    // run on CPU
    runOMPHotspot(buff, height, width, kPyramidHeight, kPyramidHeight,
            step_div_Cap, Rx, Ry, Rz);
  } else {
    // run on GPU
    gettimeofday(&start, NULL);
    runHotspot(buff, height, width, kPyramidHeight, kPyramidHeight,
            step_div_Cap, Rx, Ry, Rz, device);
    gettimeofday(&end, NULL);
  }
}

double benchmarkPCIBus(SubDomain* pS, int gpuIndex) {
  struct timeval start, end;
  double total = 0.0;
  gettimeofday(&start, NULL);
  DTYPE* devBuffer = NULL;
  int currDevice = -1;
  cudaGetDevice(&currDevice);
  if (currDevice != gpuIndex) {
    if (cudaSetDevice(gpuIndex) != cudaSuccess) {
      fprintf(stderr, "ERROR: couldn't set device to %d\n", gpuIndex);
      return -1.0;
    }
  }
  size_t size = sizeof (DTYPE) * pS->getLength(0) * pS->getLength(1);
  cudaMalloc(&devBuffer, size);
  cudaMemcpy(static_cast<void*> (devBuffer), static_cast<void*> (pS->getBuffer()),
          size, cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<void*> (pS->getBuffer()), static_cast<void*> (devBuffer),
          size, cudaMemcpyDeviceToHost);
  cudaFree(devBuffer);
  devBuffer = NULL;
  gettimeofday(&end, NULL);
  total = secondsElapsed(start, end);
  return 1 / total;
}

// TODO(den4gr)
// not sure if the destination i and j should have the border added

/*
  takes a subdomain containing results and copies it into original
  buffer, accounting for invalid ghost zone around edges
 */
void copyResultBlock2D(DTYPE* buffer, SubDomain* s, const int kBorder[2],
        const int kBufferSize[2]) {
  const int kLength[2] = {s->getLength(0) - 2 * kBorder[0],
    s->getLength(1) - 2 * kBorder[1]};
  const int kDestinationOffset[2] = {s->getOffset(0) + kBorder[0],
                                     s->getOffset(1) + kBorder[1]};
  const int kSourceOffset[2] = {kBorder[0], kBorder[1]};

  for (int i = 0; i < kLength[0]; ++i) {
    for (int j = 0; j < kLength[1]; ++j) {
      const int destI = i + kDestinationOffset[0];
      const int destJ = j + kDestinationOffset[1];
      const int srcI = i + kSourceOffset[0];
      const int srcJ = j + kSourceOffset[1];
      const int destIndex = destI * kBufferSize[1] + destJ;
      const int srcIndex = srcI * s->getLength(1) + srcJ;
      buffer[destIndex] = s->getBuffer()[srcIndex];
    }
  }
}

void copyResults2D(DTYPE* buffer, Cluster* cluster, const int kBorder[2],
        const int kBufferSize[2]) {
  if (NULL == buffer) return;

  /* get work from all parents and children in cluster */
  for (unsigned int n = 0; n < cluster->getNumNodes(); ++n) {
    Node &node = cluster->getNode(n);
    unsigned int num = node.numSubDomains();
    for (unsigned int block = 0; block < num; ++block) {
      copyResultBlock2D(buffer, node.getSubDomain(block), kBorder,
              kBufferSize);
    }

    for (unsigned int c = 0; c < node.getNumChildren(); ++c) {
      Node* child = &(node.getChild(c));
      num = child->numSubDomains();
      for (unsigned int block = 0; block < num; ++block) {
        copyResultBlock2D(buffer, child->getSubDomain(block), kBorder,
                kBufferSize);
      }
    }
  }
}

void processCPUWork(Node* machine, const int kPyramidHeight,
        const float kStepDivCap, const float kRx, const float kRy,
        const float kRz) {
  for (unsigned int task = 0; task < machine->numSubDomains(); ++task) {
    processSubDomain(kCPUIndex, machine->getSubDomain(task), kPyramidHeight,
            kStepDivCap, kRx, kRy, kRz);
  }
}

void processGPUWork(Node* machine, const int kPyramidHeight,
        const float kStepDivCap, const float kRx, const float kRy,
        const float kRz) {
  for (unsigned int gpuIndex = 0;
          gpuIndex < machine->getNumChildren();
          ++gpuIndex) {
    Node* currentDevice = &(machine->getChild(gpuIndex));
    for (unsigned int task = 0;
            task < currentDevice->numSubDomains();
            ++task) {
      processSubDomain(gpuIndex, currentDevice->getSubDomain(task),
              kPyramidHeight, kStepDivCap, kRx, kRy, kRz);
    }
  }
}

void processWork2D(Node* machine, const int kIterations, const int kPyramidHeight,
        const int kStencilSize[2], const float kStepDivCap, const float kRx,
        const float kRy, const float kRz) {
  int currentPyramidHeight = kPyramidHeight;
  const int kFirstIteration = 0;
  for (int iter = kFirstIteration; iter < kIterations; iter += kPyramidHeight) {
    if (iter + kPyramidHeight > kIterations) {
      currentPyramidHeight = kIterations - iter;
    }
    int staleBorder[3] = {kStencilSize[0] * currentPyramidHeight,
      kStencilSize[1] * currentPyramidHeight,
      kStencilSize[2] * currentPyramidHeight};
    /* The data is initially sent with the ghost zones, but since
        we actually process each subdomain interleaved with the communication,
      in receiveData, we have to update the stale cells starting with the
      first iteration. Note, this is why the number of iterations passed
      into this function should be totalIterations - pyramidHeight, due to
      the previously mentioned reason. */
    if ((machine->getRank() == 0 && iter > 0) || (machine->getRank() > 0))
      updateAllStaleData(machine, staleBorder);
    processCPUWork(machine, currentPyramidHeight, kStepDivCap, kRx, kRy, kRz);
    processGPUWork(machine, currentPyramidHeight, kStepDivCap, kRx, kRy, kRz);
  }
}

void getResultsFromCluster(Cluster* cluster) {
  /* TODO(den4gr) receives results, needs to be asynchronous */
  const bool kNoInterleavedCompute = false;
  for (unsigned int nodeRank = 1; nodeRank < cluster->getNumNodes(); ++nodeRank) {
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
        const int kIterations, const int kPyramidHeight,
        const float kStepDivCap, const float kRx, const float kRy,
        const float kRz) {
  /* TODO(den4gr) this is inefficient, need to use Bcast */
  for (unsigned int node = 1; node < cluster->getNumNodes(); ++node) {
    benchmarkNode(&(cluster->getNode(node)), data);
  }
  benchmarkMyself(&(cluster->getNode(0)), data, kIterations, kPyramidHeight,
          kStepDivCap, kRx, kRy, kRz);
}

void runDistributedHotspot(const int kMyRank, const int kNumTasks, DTYPE *data,
        const int kXMax, const int kYMax, const int kIterations,
        const int kPyramidHeight, const float kStepDivCap, const float kRx,
        const float kRy, const float kRz, const int kNumberBlocksPerDimension,
        const bool kPerformLoadBalancing, const int kDeviceConfiguration) {
  // hack because we want the compiler to give us the
  // stencil size, but we don't want to have to include
  // the cuda headers in every file, so we convert
  // it to an int array for the time-being.
  dim3 stencil_size(1, 1);
  const int kRootRank = 0;
  int new_stencil_size[3] = {stencil_size.z, stencil_size.y, stencil_size.x};
  int device_count = 0;
  double balance_time = -1.0, compute_time = -1.0, send_time = -1.0,
          receive_time = -1.0, total_time = -1.0, benchmark_time = -1.0;

  const int kBorder[2] = {new_stencil_size[0] * kPyramidHeight,
    new_stencil_size[1] * kPyramidHeight};
  Node my_work;
  Cluster* cluster = NULL;
  struct timeval send_start, send_end, rec_start, rec_end, comp_start, comp_end,
          process_start, process_end, balance_start, balance_end,
          benchmark_start, benchmark_end;

  my_work.setRank(kMyRank);
  getNumberOfChildren(&device_count);
  my_work.setNumChildren(device_count);
  
  if (kRootRank == kMyRank) {
    Decomposition decomp;
    Balancer balancer;
    // get the number of children from other nodes
    cluster = new Cluster(kNumTasks);
    cluster->getNode(kRootRank).setNumChildren(device_count);
    receiveNumberOfChildren(kNumTasks, cluster);
    printf("about to decompose.\n");
    /* perform domain decomposition */
    int numElements[2] = {kYMax, kXMax};
    decomp.decompose(data, 2, numElements, new_stencil_size, kPyramidHeight,
            kNumberBlocksPerDimension);
#ifdef DEBUG
    printDecomposition(decomp);
#endif
    printf("about to benchmark.\n");
    gettimeofday(&benchmark_start, NULL);
    benchmarkCluster(cluster, decomp.getSubDomain(0), kIterations,
            kPyramidHeight, kStepDivCap, kRx, kRy, kRz);
    gettimeofday(&benchmark_end, NULL);
    
    /* now perform the load balancing, assigning task blocks to each node */
    printf("about to balance.\n");
    gettimeofday(&balance_start, NULL);
    // passing a 0 means use cpu and gpu on all nodes
    if (kPerformLoadBalancing)
        balancer.perfBalance(*cluster, decomp, kDeviceConfiguration);
    else
        balancer.balance(*cluster, decomp, kDeviceConfiguration);
    gettimeofday(&balance_end, NULL);
    
    gettimeofday(&process_start, NULL);
    gettimeofday(&send_start, NULL);
    sendWorkToCluster(cluster);
    gettimeofday(&send_end, NULL);
    // TODO(den4gr) Is this a deep copy??
    // root's work is in the first node
    my_work = cluster->getNode(kRootRank);
    /* PROCESS ROOT NODE WORK */
    gettimeofday(&comp_start, NULL);
    processWork2D(&my_work, kIterations, kPyramidHeight, new_stencil_size,
            kStepDivCap, kRx, kRy, kRz);
    gettimeofday(&comp_end, NULL);
    
    cout << "getting results from cluster.\n";
    gettimeofday(&rec_start, NULL);
    getResultsFromCluster(cluster);
    gettimeofday(&rec_end, NULL);

    copyResults2D(data, cluster, kBorder, numElements);
    //printCluster(*cluster); // DEBUG

    gettimeofday(&process_end, NULL);
    
    delete cluster;
    cluster = NULL;
  } else {
    const bool kInterleaveProcessing = true;
    int iterations_left = kIterations;
    
    // send number of children to root
    sendNumberOfChildren(kRootRank, device_count);
    gettimeofday(&benchmark_start, NULL);
    benchmarkMyself(&my_work, NULL, kIterations, kPyramidHeight, kStepDivCap,
            kRx, kRy, kRz);
    receiveData(kRootRank, &my_work, kInterleaveProcessing, kPyramidHeight,
            kStepDivCap, kRx, kRy, kRz);
    gettimeofday(&benchmark_end, NULL);
    
    if (kInterleaveProcessing)
      iterations_left = kIterations - kPyramidHeight;
    
    gettimeofday(&comp_start, NULL);
    processWork2D(&my_work, iterations_left, kPyramidHeight, new_stencil_size,
            kStepDivCap, kRx, kRy, kRz);
    gettimeofday(&comp_end, NULL);

    // send my work back to the root
    my_work.setRank(kRootRank);
    gettimeofday(&send_start, NULL);
    sendData(&my_work);
    gettimeofday(&send_end, NULL);
    gettimeofday(&process_end, NULL);
  }
  
  total_time = secondsElapsed(process_start, process_end);
  benchmark_time = secondsElapsed(benchmark_start, benchmark_end);
  balance_time = secondsElapsed(balance_start, balance_end);
  compute_time = secondsElapsed(comp_start, comp_end);
  receive_time = secondsElapsed(rec_start, rec_end);
  send_time = secondsElapsed(send_start, send_end);
  fprintf(stdout, "*********\n");
  printf("[%d]\tbalance\t\tbenchmark\tcompute\t\treceive\t\tsend\t\ttotal\n",
          kMyRank);
  printf("[%d]\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\n",
          kMyRank, benchmark_time, balance_time, compute_time,
          send_time, receive_time, total_time);
  fprintf(stdout, "*********\n");
}
