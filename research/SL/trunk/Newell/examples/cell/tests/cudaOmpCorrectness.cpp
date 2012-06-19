/* copyright 2012 Donnie Newell */
/* test numerical correctness of distributed MPI cell version */

#define DTYPE int
#include <stdio.h>
#include "../ompCell.h"
#include "../cell.h"

DTYPE* initInput(const int kI, const int kJ, const int kK) {
  DTYPE* data = new DTYPE[kI * kJ * kK]();
  for (int z = 0; z < kI; ++z) {
    for (int y = 0; y < kJ; ++y) {
      for (int x = 0; x < kK; ++x) {
        int uidx = z * kJ * kK + y * kK + x;
        data[uidx] = uidx % 2;
      }
    }
  }
  return data;
}

void printData(const DTYPE* data, const int I, const int J, const int K) {
  for (int i = 0; i < I; ++i) {
    printf("i == %d *****************************************\n",i);
    for (int j = 0; j < J; ++j) {
      printf("row[%.2d]: ",j);
      for (int k = 0; k < K; ++k) {
        printf("%d ", data[(i * K * J) + (j * K) + k]);
      }
      printf("\n");
    }
  }
}

bool compare(DTYPE* data1, DTYPE* data2, int length) {
  for (int i = 0; i < length; ++i) {
    if (data1[i] != data2[i]) {
      printf("data1[%d]:%d != data2[%d]:%d\n", i, data1[i], i, data2[i]);
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  DTYPE* ompData = NULL, *cudaData = NULL;
  const int kDataSize = 16;
  int iterations = 1;
  int device = 0;
  int dieMin = 10;
  int dieMax = 3;
  int bornMin = 5;
  int bornMax = 8;
  bool testPass;
  printf("starting correctness test.\n");
  printf("running OpenMP version.\n");
  ompData = initInput(kDataSize, kDataSize, kDataSize);
  runOMPCell(ompData, kDataSize, kDataSize, kDataSize, iterations,
      bornMin, bornMax, dieMin, dieMax);
  runOMPCellCleanup();

  printf("running CUDA version.\n");
  cudaData = initInput(kDataSize, kDataSize, kDataSize);
  runCell(cudaData, kDataSize, kDataSize, kDataSize, iterations,
          bornMin, bornMax, dieMin, dieMax, device);
  //runCell(cudaData, kDataSize, kDataSize, kDataSize, iterations,
    //      bornMin, bornMax, dieMin, dieMax);
  runCellCleanup();
  printf("ending correctness test.\n");
  printf("cudaData:%p, ompData:%p\n", cudaData, ompData);
  testPass = compare(cudaData, ompData, kDataSize * kDataSize * kDataSize);
  if (testPass) {
    printf("Correctness passed\n");
    return 0;
  } else {
    printf("Correctness failed\n");
    printf("CUDA DATA ======================\n");
    printData(cudaData, kDataSize, kDataSize, kDataSize);
    printf("OpenMP DATA ======================\n");
    printData(ompData, kDataSize, kDataSize, kDataSize);
    return 1;
  }
}
