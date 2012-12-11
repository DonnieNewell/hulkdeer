/* copyright 2012 Donnie Newell */
/* test numerical correctness of OpenMP and CUDA cell version */

#include <stdio.h>
#include "../ompCell.h"
#include "../cell.h"
#include "gtest/gtest.h"

namespace {

// initializes the matrix to be used for input
DTYPE* initInput(const int kI, const int kJ, const int kK) {
  DTYPE* data = new DTYPE[kI * kJ * kK]();
  for (int z = 0; z < kI; ++z) {
    for (int y = 0; y < kJ; ++y) {
      for (int x = 0; x < kK; ++x) {
        int uidx = z * kJ * kK + y * kK + x;
        data[uidx] = (x) % 2;
      }
    }
  }
  return data;
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

TEST(CudaOmpTest, Equal) {
  const int kDataSize = 256;
  const int kIterations = 100;
  const int kDevice = 0;
  const int kDieMin = 15;
  const int kDieMax = 25;
  const int kBornMin = 5;
  const int kBornMax = 10;
  const int kPyramidHeight = 1;
  DTYPE* ompData = NULL, *cudaData = NULL;

  ompData = initInput(kDataSize, kDataSize, kDataSize);
  runOMPCell(ompData, kDataSize, kDataSize, kDataSize, kIterations,
              kPyramidHeight, kBornMin, kBornMax, kDieMin, kDieMax);
  runOMPCellCleanup();

  cudaData = initInput(kDataSize, kDataSize, kDataSize);
  runCell(cudaData, kDataSize, kDataSize, kDataSize, kIterations,
          kPyramidHeight, kBornMin, kBornMax, kDieMin, kDieMax, kDevice);
  runCellCleanup();

  EXPECT_TRUE(compare(cudaData, ompData, kDataSize * kDataSize * kDataSize));
}

}  // namespace

void printData(const DTYPE* data, const int I, const int J, const int K) {
  for (int i = 0; i < I; ++i) {
    printf("i == %d *****************************************\n", i);
    for (int j = J - 1; j >= 0; --j) {
      printf("row[%.2d]: ", j);
      for (int k = 0; k < K; ++k) {
        printf("%d ", data[(i * K * J) + (j * K) + k]);
      }
      printf("\n");
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
