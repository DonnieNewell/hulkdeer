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
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <omp.h>

#define SPLIT_WORK
using namespace boost;

const int kCPUIndex = -1;

// this ignores cost of exchanging Halo zones
// TODO (Donnie) account for halo zone exchange

void benchmarkMyself(Node* node, SubDomain* sub_domain, int iterations,
        const int kPyramidHeight, int bornMin, int bornMax, int dieMin,
        int dieMax) {

  // receive results for each device
  int total_devices = static_cast<int> (node->getNumChildren()) + 1;
  scoped_array<double> compute_rate(new double[2 * total_devices - 1]());
  double *comm_rate = &compute_rate[total_devices];
  const int kRootRank = 0;
  SubDomain *benchmark_block = NULL;
  int intended_device = -2;
  const int kCommIterations = 3; // multiple iterations for reliable timing
  if (sub_domain == NULL) {
    {
      /*{
        int i = 0;
        char hostname[256];
        gethostname(hostname, sizeof (hostname));
        printf("PID %d on %s ready for attach\n", getpid(), hostname);
        fflush(stdout);
        while (0 == i)
          sleep(5);
      } // */
      const int kCPUIndex = -1;
      SubDomain * blocks[3];
      for (int i = 0; i < kCommIterations; ++i) {
        blocks[i] = receiveDataFromNode(kRootRank, &intended_device);
        sendDataToNode(kRootRank, kCPUIndex, blocks[i]);
      }
      benchmark_block = blocks[0];
      for (int i = 1; i < kCommIterations; ++i)
        delete blocks[i];
    }
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
              current_pyramid_height, bornMin, bornMax, dieMin, dieMax);
    }
    gettimeofday(&end, NULL);
    total_sec = secondsElapsed(start, end);

    // numerator of 1 describes completely processing one block
    compute_rate[device_index] = 1 / total_sec;
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
    int result = MPI_Send(static_cast<void*> (compute_rate.get()),
            2 * total_devices - 1, MPI_DOUBLE, kRootRank, xWeight,
            MPI_COMM_WORLD);
    mpiCheckError(result);
  }

  // clean up
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
  int result = MPI_Recv(static_cast<void*> (&number_blocks), 1, MPI_INT, rank,
          xNumBlocks, MPI_COMM_WORLD, &stat);
  mpiCheckError(result);
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
  if (cudaSuccess != err) {
    fprintf(stderr, "getNumberOfChildren: %s\n", cudaGetErrorString(err));
    *numChildren = 0;
  }
}

void processSubDomain(int device, SubDomain *block, const int kPyramidHeight,
        int bornMin, int bornMax, int dieMin, int dieMax) {
  DTYPE* buffer = block->getBuffer();
  int depth = block->getLength(0);
  int height = block->getLength(1);
  int width = block->getLength(2);
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

void processSubDomainOuter(int device, SubDomain *block, const int kPyramidHeight,
        int bornMin, int bornMax, int dieMin, int dieMax) {
  DTYPE* buffer = block->getBuffer();
  int depth = block->getLength(0);
  int height = block->getLength(1);
  int width = block->getLength(2);
  if (-1 == device) {
    // run on CPU
    runOMPCellOuter(buffer, depth, height, width, kPyramidHeight, kPyramidHeight,
            bornMin, bornMax, dieMin, dieMax);
  } else {
    // run on GPU
    runCellOuter(buffer, depth, height, width, kPyramidHeight, kPyramidHeight,
            bornMin, bornMax, dieMin, dieMax, device);
  }
}

void processSubDomainInner(int device, SubDomain *block, const int kPyramidHeight,
        int bornMin, int bornMax, int dieMin, int dieMax) {
  DTYPE* buffer = block->getBuffer();
  int depth = block->getLength(0);
  int height = block->getLength(1);
  int width = block->getLength(2);
  if (-1 == device) {
    // run on CPU
    runOMPCellInner(buffer, depth, height, width, kPyramidHeight, kPyramidHeight,
            bornMin, bornMax, dieMin, dieMax);
  } else {
    // run on GPU
    runCellInner(buffer, depth, height, width, kPyramidHeight, kPyramidHeight,
            bornMin, bornMax, dieMin, dieMax, device);
  }
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

void processCPUWorkInner(Node* machine, const int kPyramidHeight, const int kBornMin,
        const int kBornMax, const int kDieMin, const int kDieMax) {
  for (unsigned int task = 0; task < machine->numSubDomains(); ++task) {
    processSubDomainInner(kCPUIndex, machine->getSubDomain(task), kPyramidHeight,
            kBornMin, kBornMax, kDieMin, kDieMax);
  }
}

void processCPUWorkOuter(Node* machine, const int kPyramidHeight, const int kBornMin,
        const int kBornMax, const int kDieMin, const int kDieMax) {
  for (unsigned int task = 0; task < machine->numSubDomains(); ++task) {
    processSubDomainOuter(kCPUIndex, machine->getSubDomain(task), kPyramidHeight,
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

void processGPUWorkOuter(Node* machine, const int kPyramidHeight, const int kBornMin,
        const int kBornMax, const int kDieMin, const int kDieMax) {
  for (unsigned int gpu_index = 0;
          gpu_index < machine->getNumChildren();
          ++gpu_index) {
    Node* gpu = &(machine->getChild(gpu_index));
    for (unsigned int task = 0; task < gpu->numSubDomains(); ++task) {
      processSubDomainOuter(gpu_index, gpu->getSubDomain(task), kPyramidHeight,
              kBornMin, kBornMax, kDieMin, kDieMax);
    }
  }
}

void processGPUWorkInner(Node* machine, const int kPyramidHeight, const int kBornMin,
        const int kBornMax, const int kDieMin, const int kDieMax) {
  for (unsigned int gpu_index = 0;
          gpu_index < machine->getNumChildren();
          ++gpu_index) {
    Node* gpu = &(machine->getChild(gpu_index));
    for (unsigned int task = 0; task < gpu->numSubDomains(); ++task) {
      processSubDomainInner(gpu_index, gpu->getSubDomain(task), kPyramidHeight,
              kBornMin, kBornMax, kDieMin, kDieMax);
    }
  }
}

void processWork(Node* machine, const int kIterations, const bool kInterleave,
        const int kPyramidHeight, const int kStencilSize[3],
        const int kBornMin, const int kBornMax, const int kDieMin,
        const int kDieMax) {
  int pyramid_height(kPyramidHeight), first_iter(0);
  struct timeval start, end;
  double ghost_time(0.0), compute_time(0.0);
  if (kInterleave)
    first_iter = kIterations - kPyramidHeight;
  else
    first_iter = 0;

  for (int iter = first_iter; iter < kIterations; iter += kPyramidHeight) {
    if (iter + kPyramidHeight > kIterations) {
      pyramid_height = kIterations - iter;
    }
    int stale_border[3] = {kStencilSize[0] * pyramid_height,
      kStencilSize[1] * pyramid_height,
      kStencilSize[2] * pyramid_height};
    /* The data is initially sent with the ghost zones, but since
        we actually process each subdomain interleaved with the communication,
      in receiveData, we have to update the stale cells starting with the
      first iteration. Note, this is why the number of iterations passed
      into this function should be totalIterations - pyramidHeight, due to
      the previously mentioned reason. */
    if (iter > 0) {
      gettimeofday(&start, NULL);
      updateAllStaleData(machine, stale_border);
      gettimeofday(&end, NULL);
      ghost_time += secondsElapsed(start, end);
    }
    gettimeofday(&start, NULL);
    processGPUWork(machine, pyramid_height, kBornMin, kBornMax,
            kDieMin, kDieMax);
    processCPUWork(machine, pyramid_height, kBornMin, kBornMax,
            kDieMin, kDieMax);

    gettimeofday(&end, NULL);
    compute_time += secondsElapsed(start, end);
  }
  // deallocate IO buffers and synchronization buffers
  cleanupComm(machine->numTotalSubDomains());
  printf("[%d]exchange:%10.4e\tcompute:%10.4e\n", machine->getRank(),
          ghost_time, compute_time);
}

void processWorkSplit(Node* machine, const int kIterations,
        const int kPyramidHeight, const int kStencilSize[3],
        const int kBornMin, const int kBornMax, const int kDieMin,
        const int kDieMax) {
  int pyramid_height(kPyramidHeight), first_iter(0);
  struct timeval start, end, t1, t2;
  double ghost_time(0.0), compute_time(0.0);

  first_iter = 0;

  for (int iter = first_iter; iter < kIterations; iter += kPyramidHeight) {
    if (iter + kPyramidHeight > kIterations) {
      pyramid_height = kIterations - iter;
    }
    int stale_border[3] = {kStencilSize[0] * pyramid_height,
      kStencilSize[1] * pyramid_height,
      kStencilSize[2] * pyramid_height};
    int num_messages_sent = 0;
    int num_threads(0), thread_id(0);
    /* The data is initially sent with the ghost zones, but since
       we actually process each subdomain interleaved with the communication,
       in receiveData, we have to update the stale cells starting with the
       first iteration. Note, this is why the number of iterations passed
       into this function should be totalIterations - pyramidHeight, due to
       the previously mentioned reason. */
    //printf("processing outer");

    // *** PROCESS GHOST ZONE ***
    gettimeofday(&start, NULL);
    //gettimeofday(&t1, NULL);
    processGPUWorkOuter(machine, pyramid_height, kBornMin, kBornMax,
            kDieMin, kDieMax);
    //gettimeofday(&t2, NULL);
    //printf("GPU iter took:%f sec\n", secondsElapsed(t1, t2));
    //gettimeofday(&t1, NULL);
    processCPUWorkOuter(machine, pyramid_height, kBornMin, kBornMax,
            kDieMin, kDieMax);
    //gettimeofday(&t2, NULL);
    //printf("CPU iter took:%f sec\n", secondsElapsed(t1, t2));
    gettimeofday(&end, NULL);
    compute_time += secondsElapsed(start, end);

    // *** START ASYNC GHOST ZONE UPDATE ***
    //gettimeofday(&end, NULL);
    //compute_time += secondsElapsed(start, end);
    gettimeofday(&start, NULL);
    updateStart(machine, stale_border, &num_messages_sent);
    gettimeofday(&end, NULL);
    ghost_time += secondsElapsed(start, end);

    // *** PROCESS INNER DATA ***
    gettimeofday(&start, NULL);
    //gettimeofday(&t1, NULL);
    processGPUWorkInner(machine, pyramid_height, kBornMin, kBornMax,
            kDieMin, kDieMax);
    //gettimeofday(&t2, NULL);
    //printf("GPU iter took:%f sec\n", secondsElapsed(t1, t2));
    //gettimeofday(&t1, NULL);
    processCPUWorkInner(machine, pyramid_height, kBornMin, kBornMax,
            kDieMin, kDieMax);
    //gettimeofday(&t2, NULL);
    //printf("CPU iter took:%f sec\n", secondsElapsed(t1, t2));
    gettimeofday(&end, NULL);
    compute_time += secondsElapsed(start, end);

    // *** FINISH ASYNC GHOST ZONE UPDATE ***
    gettimeofday(&start, NULL);
    updateFinish(machine, stale_border, num_messages_sent);
    gettimeofday(&end, NULL);
    ghost_time += secondsElapsed(start, end);
  }

  // deallocate IO buffers and synchronization buffers
  cleanupComm(machine->numTotalSubDomains());

  printf("[%d] exchange:%10.4e\tcompute:%10.4e\n", machine->getRank(),
          ghost_time, compute_time);
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

void benchmarkCluster(Cluster* cluster, SubDomain* data, const int kIterations,
        const int kPyramidHeight, const int kBornMin, const int kBornMax,
        const int kDieMin, const int kDieMax) {
  /* TODO(den4gr) this is inefficient, need to use Bcast */
  for (unsigned int node = 1; node < cluster->getNumNodes(); ++node) {
    benchmarkNode(&(cluster->getNode(node)), data);
  }
  benchmarkMyself(&(cluster->getNode(0)), data, kIterations, kPyramidHeight,
          kBornMin, kBornMax, kDieMin, kDieMax);
}

void runDistributedCell(const int kMyRank, const int kNumTasks, DTYPE *data,
        const int kXMax, const int kYMax, const int kZMax,
        const int kIterations, const int kPyramidHeight, const int kBornMin,
        const int kBornMax, const int kDieMin, const int kDieMax,
        const int kNumberBlocksPerDimension, const bool kPerformLoadBalancing,
        const int kDeviceConfiguration) {

  // hack because we want the compiler to give us the
  // stencil size, but we don't want to have to include
  // the cuda headers in every file, so we convert
  // it to an int array for the time-being.
  dim3 stencil_size(1, 1, 1);
  const int kRootRank = 0;
  int new_stencil_size[3] = {stencil_size.z, stencil_size.y, stencil_size.x};
  int device_count = 0;
  double total_time = -1.0;

  const int kBorder[3] = {new_stencil_size[0] * kPyramidHeight,
    new_stencil_size[1] * kPyramidHeight,
    new_stencil_size[2] * kPyramidHeight};
  Node my_work;
  struct timeval send_start, send_end, rec_start, rec_end, comp_start, comp_end,
          process_start, process_end, balance_start, balance_end,
          benchmark_start, benchmark_end;

  my_work.setRank(kMyRank);
  getNumberOfChildren(&device_count);
  my_work.setNumChildren(device_count);
  printf("node:%d has %d children\n", kMyRank, device_count);

  if (kRootRank == kMyRank) {
    Decomposition decomp;
    Balancer balancer;

    // get the number of children from other nodes
    boost::scoped_ptr<Cluster> cluster(new Cluster(kNumTasks));
    cluster->getNode(kRootRank).setNumChildren(device_count);
    receiveNumberOfChildren(kNumTasks, cluster.get());
    /* perform domain decomposition */
    const int kNumElements[3] = {kZMax, kYMax, kXMax};
    decomp.decompose(data, 3, kNumElements, new_stencil_size, kPyramidHeight,
            kNumberBlocksPerDimension);
#ifdef DEBUG
    printDecomposition(decomp);
#endif
    gettimeofday(&benchmark_start, NULL);
    benchmarkCluster(cluster.get(), decomp.getSubDomain(0), kIterations,
            kPyramidHeight, kBornMin, kBornMax, kDieMin, kDieMax);
    gettimeofday(&benchmark_end, NULL);

    /* now perform the load balancing, assigning task blocks to each node */
    gettimeofday(&balance_start, NULL);

    //printf("load balancing...\n");
    if (kPerformLoadBalancing)
      balancer.perfBalance(*cluster, decomp, kDeviceConfiguration);
    else
      balancer.balance(*cluster, decomp, kDeviceConfiguration);
    gettimeofday(&balance_end, NULL);

    printCluster(*cluster); // DEBUG
    gettimeofday(&process_start, NULL);
    //printf("sending work to cluster...\n");
    gettimeofday(&send_start, NULL);
    sendWorkToCluster(cluster.get());
    gettimeofday(&send_end, NULL);
    // root's work is in the first node
    my_work = cluster->getNode(kRootRank);
    /* PROCESS ROOT NODE WORK */
    //printf("[%d]\t%10s\t%10s\t%10s\n", kMyRank, "update", "compute", "total");
    gettimeofday(&comp_start, NULL);
    //printf("process work...\n");
#ifdef SPLIT_WORK
    processWorkSplit(&my_work, kIterations, kPyramidHeight, new_stencil_size,
            kBornMin, kBornMax, kDieMin, kDieMax);
#else
    processWork(&my_work, kIterations, false, kPyramidHeight, new_stencil_size,
            kBornMin, kBornMax, kDieMin, kDieMax);
#endif
    gettimeofday(&comp_end, NULL);

    //printf("get results...\n");
    gettimeofday(&rec_start, NULL);
    getResultsFromCluster(cluster.get());
    gettimeofday(&rec_end, NULL);

    copy_results(data, cluster.get(), kBorder, kNumElements);
    gettimeofday(&process_end, NULL);
    total_time = secondsElapsed(process_start, process_end);
    printf("[%d] cuda_memcpy_time:%f  memcpy_time:%f\n", kMyRank,
            getGpuMemcpyTime(), getMemcpyTime());
    printf("Total: %10.4e\n", total_time);
  } else {
    const bool kInterleaveProcessing = false;

    int iterations_left = kIterations;

    sendNumberOfChildren(kRootRank, device_count);
    gettimeofday(&benchmark_start, NULL);
    benchmarkMyself(&my_work, NULL, kIterations, kPyramidHeight, kBornMin,
            kBornMax, kDieMin, kDieMax);
    gettimeofday(&benchmark_end, NULL);

    gettimeofday(&process_start, NULL);
    gettimeofday(&rec_start, NULL);
    receiveData(kRootRank, &my_work, kInterleaveProcessing, kPyramidHeight,
            kBornMin, kBornMax, kDieMin, kDieMax);
    gettimeofday(&rec_end, NULL);

    gettimeofday(&comp_start, NULL);
#ifdef SPLIT_WORK
    processWorkSplit(&my_work, iterations_left, kPyramidHeight,
            new_stencil_size, kBornMin, kBornMax, kDieMin, kDieMax);
#else
    processWork(&my_work, iterations_left, false, kPyramidHeight,
            new_stencil_size, kBornMin, kBornMax, kDieMin, kDieMax);
#endif
    gettimeofday(&comp_end, NULL);

    // send my work back to the root
    my_work.setRank(kRootRank);
    gettimeofday(&send_start, NULL);
    sendData(&my_work);
    gettimeofday(&send_end, NULL);
    gettimeofday(&process_end, NULL);
  }

  /*benchmark_time = secondsElapsed(benchmark_start, benchmark_end);
  balance_time = secondsElapsed(balance_start, balance_end);
  compute_time = secondsElapsed(comp_start, comp_end);
  receive_time = secondsElapsed(rec_start, rec_end);
  send_time = secondsElapsed(send_start, send_end);  // */
}
