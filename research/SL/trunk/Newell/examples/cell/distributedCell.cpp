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
#include "./cell.h"
#include "./ompCell.h"
#include "./distributedCell.h"

const int kCPUIndex = -1;

// this ignores cost of exchanging Halo zones
// TODO (Donnie) account for halo zone exchange

void benchmarkMyself(Node* node, SubDomain* sub_domain, int iterations,
        const int kPyramidHeight, int bornMin, int bornMax, int dieMin,
        int dieMax) {
  // receive results for each device
  int total_devices = static_cast<int> (node->getNumChildren()) + 1;
  double *task_per_sec = new double[2 * total_devices - 1]();
  double *edge_weight = &task_per_sec[total_devices];
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
      processSubDomain(device_index - 1, benchmark_block, current_pyramid_height,
              bornMin, bornMax, dieMin, dieMax);
    }
    gettimeofday(&end, NULL);
    total_sec = secondsElapsed(start, end);
    printf("[%d]benchmarking device %d. time: %f seconds.\n",
            node->getRank(), device_index - 1, total_sec);

    task_per_sec[device_index] = iterations / total_sec;
    if (device_index == 0) {
      node->setWeight(task_per_sec[device_index]);
      node->setEdgeWeight(benchmarkSubDomainCopy(benchmark_block));
    } else {
      node->getChild(device_index - 1).setWeight(task_per_sec[device_index]);
      edge_weight[device_index - 1] = benchmarkPCIBus(benchmark_block, device_index - 1);
      node->getChild(device_index - 1).setEdgeWeight(edge_weight[device_index - 1]);
    }
  }

  if (NULL == sub_domain) {
    // send the result back to the host
    MPI_Send(static_cast<void*> (task_per_sec), 2 * total_devices - 1, MPI_DOUBLE, 0,
            xWeight, MPI_COMM_WORLD);
  }

  // clean up
  delete [] task_per_sec;
  task_per_sec = edge_weight = NULL;
  if (sub_domain == NULL) {
    delete benchmark_block;
    benchmark_block = NULL;
  }
}

void receiveData(int rank, Node* n, bool processNow, int pyramidHeight,
        int bornMin, int bornMax, int dieMin, int dieMax) {
  // receive number of task blocks that will be sent
  int number_blocks = 0;
  MPI_Status stat;
  MPI_Recv(static_cast<void*> (&number_blocks), 1, MPI_INT, rank, xNumBlocks,
          MPI_COMM_WORLD, &stat);
  struct timeval start, end;
  double receive_time = 0.0, process_time = 0.0;
  for (int block_index = 0; block_index < number_blocks; ++block_index) {
    SubDomain* s = NULL;
    int device = -2;
    gettimeofday(&start, NULL);
    s = receiveDataFromNode(rank, &device);
    gettimeofday(&end, NULL);
    receive_time += secondsElapsed(start, end);
    if (processNow) {
      gettimeofday(&start, NULL);
      processSubDomain(device, s, pyramidHeight, bornMin, bornMax,
              dieMin, dieMax);
      gettimeofday(&end, NULL);
      process_time += secondsElapsed(start, end);
    }
    if (-1 == device) // add block to cpu queue
      n->addSubDomain(s);
    else // add block to gpu queue
      n->getChild(device).addSubDomain(s);
  }
  //runCellCleanup();
}

void getNumberOfChildren(int* numChildren) {
  /* check to see how many NVIDIA GPU'S ARE AVAILABLE */
  cudaError_t err = cudaGetDeviceCount(numChildren);
  if (cudaSuccess == cudaErrorNoDevice) {
    *numChildren = 0;
  } else if (cudaSuccess != err) {
    //fprintf(stderr, "error detecting cuda-enabled devices\n");
    *numChildren = 0;
  }
}

void sendNumberOfChildren(const int dest_rank, const int numChildren) {
  MPI_Request req;
  int sendNumChildrenBuffer = numChildren;
  MPI_Isend(static_cast<void*> (&sendNumChildrenBuffer), 1, MPI_INT, dest_rank,
          xChildren, MPI_COMM_WORLD, &req);
  MPI_Waitall(1, &req, MPI_STATUSES_IGNORE);
}

void processSubDomain(int device, SubDomain *block, const int kPyramidHeight,
        int bornMin, int bornMax, int dieMin, int dieMax) {

  DTYPE* buffer = block->getBuffer();
  int depth = block->getLength(0);
  int height = block->getLength(1);
  int width = block->getLength(2);
  struct timeval start, end;
  if (-1 == device) {
    // run on CPU
    runOMPCell(buffer, depth, height, width, kPyramidHeight, kPyramidHeight,
            bornMin, bornMax, dieMin, dieMax);
  } else {
    // run on GPU
    runCell(buffer, depth, height, width, kPyramidHeight, kPyramidHeight,
            bornMin, bornMax, dieMin, dieMax, device);
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

  DTYPE* destination = new DTYPE[size]();
  for (int iteration = 0; iteration < kNumberIterations; ++iteration) {
    memcpy(destination, sub_domain->getBuffer(), size * kTypeSize);
    memcpy(sub_domain->getBuffer(), destination, size * kTypeSize);
  }
  delete [] destination;
  gettimeofday(&stop, NULL);
  time_elapsed = secondsElapsed(start, stop) / kNumberIterations;
  return 1 / time_elapsed;
}

double benchmarkPCIBus(SubDomain* sub_domain, int gpuIndex) {
  struct timeval start, end;
  double total = 0.0;
  const int kNumberIterations = 20;
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
  size_t size = sizeof (DTYPE) * sub_domain->getLength(0) *
          sub_domain->getLength(1) * sub_domain->getLength(2);
  cudaMalloc(&devBuffer, size);
  for (int iteration = 0; iteration < kNumberIterations; ++iteration) {
    cudaMemcpy(static_cast<void*> (devBuffer), static_cast<void*> (sub_domain->getBuffer()),
            size, cudaMemcpyHostToDevice);
    cudaMemcpy(static_cast<void*> (sub_domain->getBuffer()), static_cast<void*> (devBuffer),
            size, cudaMemcpyDeviceToHost);
  }
  cudaFree(devBuffer);
  devBuffer = NULL;
  gettimeofday(&end, NULL);
  total = secondsElapsed(start, end) / kNumberIterations;
  return 1 / total;
}
// TODO(den4gr)

/*
  takes a subdomain containing results and copies it into original
  buffer, accounting for invalid ghost zone around edges
 */
void copy_result_block(DTYPE* buffer, SubDomain* s, const int kBorder[3],
        const int kBuffSize[3]) {
  const int kLength[3] = {s->getLength(0) - 2 * kBorder[0],
    s->getLength(1) - 2 * kBorder[1],
    s->getLength(2) - 2 * kBorder[2]};
  const int kDestinationOffset[3] = {s->getOffset(0) + kBorder[0],
    s->getOffset(1) + kBorder[1],
    s->getOffset(2) + kBorder[2]};
  const int kSourceOffset[3] = {kBorder[0], kBorder[1], kBorder[2]};
  for (int i = 0; i < kLength[0]; ++i) {
    for (int j = 0; j < kLength[1]; ++j) {
      for (int k = 0; k < kLength[2]; ++k) {
        int destI = kDestinationOffset[0] + i;
        int destJ = kDestinationOffset[1] + j;
        int destK = kDestinationOffset[2] + k;
        int srcI = kSourceOffset[0] + i;
        int srcJ = kSourceOffset[1] + j;
        int srcK = kSourceOffset[2] + k;
        int destIndex = (destI * kBuffSize[1] + destJ) * kBuffSize[2] + destK;
        int srcIndex = (srcI * s->getLength(1) + srcJ) * s->getLength(2) + srcK;
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
    for (unsigned int block = 0; block < num; ++block) {
      copy_result_block(buffer, node.getSubDomain(block), kBorder, kBufferSize);
    }

    for (unsigned int c = 0; c < node.getNumChildren(); ++c) {
      Node* child = &(node.getChild(c));
      num = child->numSubDomains();

      for (unsigned int block = 0; block < num; ++block) {
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
  for (unsigned int gpu_index = 0;
          gpu_index < machine->getNumChildren();
          ++gpu_index) {
    Node* gpu = &(machine->getChild(gpu_index));
    for (unsigned int task = 0; task < gpu->numSubDomains(); ++task) {
      processSubDomain(gpu_index, gpu->getSubDomain(task), kPyramidHeight,
              kBornMin, kBornMax, kDieMin, kDieMax);
    }
  }
}

void processWork(Node* machine, const int kIterations, const int kPyramidHeight,
        const int kStencilSize[3], const int kBornMin,
        const int kBornMax, const int kDieMin, const int kDieMax) {
  //fprintf(stderr, "[%d] processWork(iterations:%d)\n", machine->getRank(),
  //      kIterations);
  int current_pyramid_height = kPyramidHeight;
  struct timeval start, end;
  double ghost_time(0.0), compute_time(0.0);
  const int kFirstIteration = 0;
  for (int iter = kFirstIteration; iter < kIterations; iter += kPyramidHeight) {
    if (iter + kPyramidHeight > kIterations) {
      current_pyramid_height = kIterations - iter;
    }
    int stale_border[3] = {kStencilSize[0] * current_pyramid_height,
      kStencilSize[1] * current_pyramid_height,
      kStencilSize[2] * current_pyramid_height};
    /* The data is initially sent with the ghost zones, but since
        we actually process each subdomain interleaved with the communication,
      in receiveData, we have to update the stale cells starting with the
      first iteration. Note, this is why the number of iterations passed
      into this function should be totalIterations - pyramidHeight, due to
      the previously mentioned reason. */
    if ((machine->getRank() == 0 && iter > 0) || (machine->getRank() > 0)) {
      //  fprintf(stderr, "[%d] *** updating stale data.\n", machine->getRank());
      gettimeofday(&start, NULL);
      updateAllStaleData(machine, stale_border);
      gettimeofday(&end, NULL);
    }
    ghost_time += secondsElapsed(start, end);
    // fprintf(stderr, "[%d] *** processing iter:%d.\n", machine->getRank(), iter);
    gettimeofday(&start, NULL);
    processCPUWork(machine, current_pyramid_height, kBornMin, kBornMax,
            kDieMin, kDieMax);
    processGPUWork(machine, current_pyramid_height, kBornMin, kBornMax,
            kDieMin, kDieMax);
    gettimeofday(&end, NULL);
    compute_time += secondsElapsed(start, end);
  }
  // deallocate IO buffers and synchronization buffers
  cleanupComm(machine->numTotalSubDomains());
  printf("[%d]processWork: update: %f seconds compute: %f seconds\n",
          machine->getRank(), ghost_time, compute_time);
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
        const int kIterations, const int kPyramidHeight,
        const int kBornMin, const int kBornMax,
        const int kDieMin, const int kDieMax) {
  /* TODO(den4gr) this is inefficient, need to use Bcast */
  for (unsigned int node = 1; node < cluster->getNumNodes(); ++node) {
    benchmarkNode(&(cluster->getNode(node)), data);
  }
  benchmarkMyself(&(cluster->getNode(0)), data, kIterations, kPyramidHeight,
          kBornMin, kBornMax, kDieMin, kDieMax);
}

void runDistributedCell(const int kMyRank, const int kNumTasks, DTYPE *data,
        const int kXMax, const int kYMax, const int kZMax, const int kIterations,
        const int kPyramidHeight, const int kBornMin, const int kBornMax,
        const int kDieMin, const int kDieMax, const int kNumberBlocksPerDimension,
        const bool kPerformLoadBalancing, const int kDeviceConfiguration) {
  // hack because we want the compiler to give us the
  // stencil size, but we don't want to have to include
  // the cuda headers in every file, so we convert
  // it to an int array for the time-being.
  dim3 stencil_size(1, 1, 1);
  int new_stencil_size[3] = {stencil_size.z, stencil_size.y, stencil_size.x};
  int device_count = 0;
  double balance_time = -1.0, compute_time = -1.0, send_time = -1.0,
          receive_time = -1.0, total_time = -1.0, benchmark_time = -1.0;

  const int kBorder[3] = {new_stencil_size[0] * kPyramidHeight,
    new_stencil_size[1] * kPyramidHeight,
    new_stencil_size[2] * kPyramidHeight};
  Node my_work;
  Cluster* cluster = NULL;
  struct timeval send_start, send_end, rec_start, rec_end, comp_start, comp_end,
          process_start, process_end, balance_start, balance_end,
          benchmark_start, benchmark_end;

  my_work.setRank(kMyRank);
  getNumberOfChildren(&device_count);
  my_work.setNumChildren(device_count);

  if (0 == kMyRank) {
    Decomposition decomp;
    Balancer lb;

    // get the number of children from other nodes
    cluster = new Cluster(kNumTasks);
    cluster->getNode(0).setNumChildren(device_count);
    receiveNumberOfChildren(kNumTasks, cluster);
    /* perform domain decomposition */
    int numElements[3] = {kZMax, kYMax, kXMax};
    decomp.decompose(data, 3, numElements, new_stencil_size, kPyramidHeight,
            kNumberBlocksPerDimension);
#ifdef DEBUG
    printDecomposition(decomp);
#endif
    gettimeofday(&benchmark_start, NULL);
    benchmarkCluster(cluster, decomp.getSubDomain(0), kIterations,
            kPyramidHeight, kBornMin, kBornMax, kDieMin, kDieMax);
    gettimeofday(&benchmark_end, NULL);

    /* now perform the load balancing, assigning task blocks to each node */
    gettimeofday(&balance_start, NULL);

    if (kPerformLoadBalancing)
      lb.perfBalance(*cluster, decomp, kDeviceConfiguration);
    else
      lb.balance(*cluster, decomp, kDeviceConfiguration);
    gettimeofday(&balance_end, NULL);

    printCluster(*cluster); // DEBUG
    gettimeofday(&process_start, NULL);
    gettimeofday(&send_start, NULL);
    sendWorkToCluster(cluster);
    gettimeofday(&send_end, NULL);
    // root's work is in the first node
    my_work = cluster->getNode(0);
    /* PROCESS ROOT NODE WORK */
    gettimeofday(&comp_start, NULL);
    processWork(&my_work, kIterations, kPyramidHeight, new_stencil_size,
            kBornMin, kBornMax, kDieMin, kDieMax);
    gettimeofday(&comp_end, NULL);

    gettimeofday(&rec_start, NULL);
    getResultsFromCluster(cluster);
    gettimeofday(&rec_end, NULL);

    copy_results(data, cluster, kBorder, numElements);
    gettimeofday(&process_end, NULL);
    delete cluster;
    cluster = NULL;
  } else {
    const bool kInterleaveProcessing = true;
    const int kRootNodeIndex = 0;
    int iterations_left = kIterations;

    sendNumberOfChildren(kRootNodeIndex, device_count);
    gettimeofday(&benchmark_start, NULL);
    benchmarkMyself(&my_work, NULL, kIterations, kPyramidHeight, kBornMin,
            kBornMax, kDieMin, kDieMax);
    gettimeofday(&benchmark_end, NULL);

    gettimeofday(&process_start, NULL);
    gettimeofday(&rec_start, NULL);
    receiveData(kRootNodeIndex, &my_work, kInterleaveProcessing, kPyramidHeight,
            kBornMin, kBornMax, kDieMin, kDieMax);
    gettimeofday(&rec_end, NULL);

    if (kInterleaveProcessing)
      iterations_left = kIterations - kPyramidHeight;

    gettimeofday(&comp_start, NULL);
    processWork(&my_work, iterations_left, kPyramidHeight, new_stencil_size,
            kBornMin, kBornMax, kDieMin, kDieMax);
    gettimeofday(&comp_end, NULL);

    // send my work back to the root
    my_work.setRank(kRootNodeIndex);
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
