/* copyright 2012 Donnie Newell */
/* test numerical correctness of distributed MPI cell version */

#define DTYPE int
#include <stdio.h>
#include "mpi.h"
#include "../ompCell.h"
#include "../distributedCell.h"

DTYPE* initInput(int i, int j, int k) {
  DTYPE* data = new DTYPE[i * j * k]();
  for (int x = 0; x < i * j * k; ++x) {
    data[x] = 1;
  }
  return data;
}

bool compare(DTYPE* data1, DTYPE* data2, int length) {
  for (int i = 0; i < length; ++i) {
    if (data1[i] != data2[i]) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  DTYPE* ompData = NULL, *mpiData = NULL;
  const int kDataSize = 32;
  int iterations = 2;
  int dieMin = 10;
  int dieMax = 3;
  int bornMin = 5;
  int bornMax = 8;
  bool testPass;
  int returnCode, numTasks, rank;
  returnCode = MPI_Init(&argc, &argv);
  if (returnCode != MPI_SUCCESS) {
    fprintf(stderr, "Error initializing MPI.\n");
    MPI_Abort(MPI_COMM_WORLD, returnCode);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("starting correctness test.\n");
  if (0 == rank) {
    printf("running OpenMP version.\n");
    ompData = initInput(kDataSize, kDataSize, kDataSize);
    runOMPCell(ompData, kDataSize, kDataSize, kDataSize, iterations,
        bornMin, bornMax, dieMin, dieMax);
    runOMPCellCleanup();

    mpiData = initInput(kDataSize, kDataSize, kDataSize);
    printf("running MPI version.\n");
  }
  runDistributedCell(rank, numTasks, mpiData, kDataSize, kDataSize, kDataSize,
      iterations, bornMin, bornMax, dieMin, dieMax);
  printf("ending correctness test.\n");
  if (0 == rank) {
    printf("mpiData:%p, ompData:%p\n", mpiData, ompData);
    testPass = compare(mpiData, ompData, kDataSize * kDataSize * kDataSize);
    if (testPass) {
      printf("Correctness passed\n");
      MPI_Finalize();
      return 0;
    } else {
      printf("Correctness failed\n");
      MPI_Finalize();
      return 1;
    }
  }
  MPI_Finalize();
}
