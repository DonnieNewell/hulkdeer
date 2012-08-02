/* -*- Mode: C ; indent-tabs-mode: nil ; c-file-style: "stroustrup" -*-
 * Copyright 2012 University of Virginia
   CS 6620 - Compilers
   Stencil App Language Project
Authors: Greg Faust, Sal Valente, derived from code by Jiayuan Meng


File:   cell-main.cpp     Contains a main routine to drive the pathfinder example.
 */

#include <stdlib.h>
#include <stdio.h>
#ifndef WIN32
#include <sys/time.h>
#else
#include<time.h>
#endif
#include <fstream>
#include "distributedCell.h"
#include "mpi.h"


/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

void printUsage();

int J, K, L;
int* data;
int** space2D;
int*** space3D;
#define M_SEED 9
int pyramid_height;
int timesteps;
int number_blocks_per_dimension;
bool perform_load_balancing;
int device_configuration;
// #define BENCH_PRINT

void
initData(int length[3]) {
#ifdef DEBUG
  fprintf(stderr, "initializing data only.\n");
#endif
  J = length[0];
  K = length[1];
  L = length[2];
#ifdef DEBUG
  fprintf(stderr, "allocating data[%d][%d][%d].\n", L, K, J);
#endif
  data = new int[J * K * L];
#ifdef DEBUG
  fprintf(stderr, "allocating space2D.\n");
#endif
  space2D = new int*[J * K];
#ifdef DEBUG
  fprintf(stderr, "allocating space3D.\n");
#endif
  space3D = new int**[J];
#ifdef DEBUG
  fprintf(stderr, "initializing space2D only.\n");
#endif
  for (int n = 0; n < (J * K); n++)
    space2D[n] = data + (L * n);
#ifdef DEBUG
  fprintf(stderr, "initializing space3D only.\n");
#endif
  for (int n = 0; n < J; n++)
    space3D[n] = space2D + (K * n);
}

void
init(int argc, char** argv, const int kMyRank) {
#ifdef DEBUG
  fprintf(stderr, "initializing data for root node.\n");
#endif
  if (argc == 9) {
    J = atoi(argv[1]);
    K = atoi(argv[2]);
    L = atoi(argv[3]);
    timesteps = atoi(argv[4]);
    pyramid_height = atoi(argv[5]);
    number_blocks_per_dimension = atoi(argv[6]);
    perform_load_balancing = (atoi(argv[7]) != 0) ? true : false;
    device_configuration = atoi(argv[8]);
  } else {
    printUsage();
    exit(0);
  }
  if (kMyRank == 0) {
    data = new int[J * K * L];
    space2D = new int*[J * K];
    space3D = new int**[J];
    for (int n = 0; n < (J * K); n++)
      space2D[n] = data + (L * n);
    for (int n = 0; n < J; n++)
      space3D[n] = space2D + (K * n);

    unsigned int seed = M_SEED;
    for (int i = 0; i < (J * K * L); i++)
      data[i] = rand_r(&seed) % 2;
  }
}

void printResults(int* data, int J, int K, int L) {
  int total = J * K * L;
  for (int n = 0; n < total; n++) {
    printf("%d ", data[n]);
  }
  printf("\n");
}

int bornMin = 5, bornMax = 8;
int dieMax = 3, dieMin = 10;

int main(int argc, char** argv) {
  int return_code = 0, num_tasks = 0, my_rank = 0;
  return_code = MPI_Init(&argc, &argv);
  if (return_code != MPI_SUCCESS) {
    fprintf(stderr, "Error initializing MPI.\n");
    MPI_Abort(MPI_COMM_WORLD, return_code);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  init(argc, argv, my_rank); // initialize data

#ifdef STATISTICS
  for (int i = 40; i <= J; i += 20) {
    // Set iteration count so that kernel is called at least 30 times.
    // The maximum pyramid height is 3, so iterations = 90.
    runDistributedCell(my_rank, num_tasks, data, i, i, i, 90, pyramid_height,
            bornMin, bornMax, dieMin, dieMax, number_blocks_per_dimension,
            perform_load_balancing, device_configuration);
  }
#else
  runDistributedCell(my_rank, num_tasks, data, J, K, L, timesteps, pyramid_height,
          bornMin, bornMax, dieMin, dieMax, number_blocks_per_dimension,
          perform_load_balancing, device_configuration);
#endif

#ifdef BENCH_PRINT
  if (my_rank == 0) {
    printResults(data, J, K, L);
  }
#endif
  MPI_Finalize();
  delete [] data;
  delete [] space2D;
  delete [] space3D;

  return 0;
}

void printUsage() {
  printf("./distributedCell [dim1] [dim2] [dim3] [timesteps] [pyramid_height] [blocks_per_dimension] [load_balance] [device_configuration]\n");
}
