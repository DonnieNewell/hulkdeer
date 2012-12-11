/* 
 * File:   aggregate.cpp
 * Author: den4gr
 *
 * Created on October 6, 2012, 12:11 PM
 */

#include <stdlib.h>
#include <iostream>
#include "../Decomposition.h"
#include "gtest/gtest.h"

namespace {

/*
 * Simple C++ Test Suite
 */

// initializes the blessed block to be used for comparison
void initSuperBlock(const DTYPE kValue1, const DTYPE kValue2, const int kFirstI,
        const int kTotalI, const int kJ, const int kK, DTYPE* data) {
  for (int z = 0; z < kFirstI; ++z) {
    for (int y = 0; y < kJ; ++y) {
      for (int x = 0; x < kK; ++x) {
        int uidx = z * kJ * kK + y * kK + x;
        data[uidx] = kValue1;
      }
    }
  }
  for (int z = kFirstI; z < kTotalI; ++z) {
    for (int y = 0; y < kJ; ++y) {
      for (int x = 0; x < kK; ++x) {
        int uidx = z * kJ * kK + y * kK + x;
        data[uidx] = kValue2;
      }
    }
  }
}

void printData(const DTYPE* data, const int I, const int J, const int K) {
  for (int i = 0; i < I; ++i) {
    printf("i == %d *****************************************\n", i);
    for (int j = 0; j < J; ++j) {
      printf("row[%d]:\t", j);
      for (int k = 0; k < K; ++k) {
        printf("%d ", data[(i * K * J) + (j * K) + k]);
      }
      printf("\n");
    }
  }
}

// initializes the matrix to be used for input
void initSubBlock(const DTYPE kValue, const int kI, const int kJ, const int kK,
        DTYPE* data) {
  for (int z = 0; z < kI; ++z) {
    for (int y = 0; y < kJ; ++y) {
      for (int x = 0; x < kK; ++x) {
        int uidx = z * kJ * kK + y * kK + x;
        data[uidx] = kValue;
      }
    }
  }
}

bool compare(DTYPE* data1, DTYPE* data2, const int kDepth, const int kHeight,
        const int kWidth) {
  for (int i = 0; i < kDepth * kHeight * kWidth; ++i) {
    if (data1[i] != data2[i]) {
      printf("data1[%d]:%d != data2[%d]:%d\n", i, data1[i], i, data2[i]);
      printf("11111111 * * * * * * * * * * * * * * *\n");
      printData(data1, kDepth, kHeight, kWidth);
      printf("22222222 * * * * * * * * * * * * * * *\n");
      printData(data2, kDepth, kHeight, kWidth);
      return false;
    }
  }
  return true;
}

TEST(AggregateTest, CombineTwoBlocks) {
  const int kDataSize = 4;
  const int kPyramidHeight = 1;
  const int kStencilSize[] = {1, 1, 1};
  const int kBorder[] = { kStencilSize[0] * kPyramidHeight,
                          kStencilSize[1] * kPyramidHeight,
                          kStencilSize[2] * kPyramidHeight };
  int id[] = {0, 0, 0};
  const int kValue1 = 1, kValue2 = 2;
  int fake_neighbors[26] = {0};
  const int kGridDim[] = {2, 1, 1};
  // create test block and two separate sub-blocks
  SubDomain block_0(id, -kBorder[0], (2 * kDataSize) + 2 * kBorder[0], 0,
          kDataSize, 0, kDataSize, kGridDim[0], kGridDim[1], kGridDim[2],
          fake_neighbors);
  SubDomain* block_1 = new SubDomain(id, -kBorder[0],
          kDataSize + 2 * kBorder[0], 0, kDataSize, 0, kDataSize, kGridDim[0],
          kGridDim[1], kGridDim[2], fake_neighbors);
  block_1->setBorder(0, kBorder[0]);
  
  id[0] = 1;
  SubDomain* block_2 = new SubDomain(id, kDataSize - kBorder[0],
          kDataSize + 2 * kBorder[0], 0, kDataSize, 0, kDataSize,
          kGridDim[0], kGridDim[1], kGridDim[2], fake_neighbors);
  block_2->setBorder(0, kBorder[0]);
  
  // initialize block values
  initSuperBlock(kValue1, kValue2, block_0.getLength(0) / 2,
          block_0.getLength(0), block_0.getLength(1), block_0.getLength(2),
          block_0.getBuffer());
  initSubBlock(kValue1, block_1->getLength(0), block_1->getLength(1),
          block_1->getLength(2), block_1->getBuffer());
  initSubBlock(kValue2, block_2->getLength(0), block_2->getLength(1),
          block_2->getLength(2), block_2->getBuffer());
  
  Decomposition decomposition;
  decomposition.addSubDomain(block_1);
  decomposition.addSubDomain(block_2);
  block_1 = NULL;
  block_2 = NULL;
  SubDomain* aggregate = decomposition.getAggregate3D(2);
  EXPECT_TRUE(compare(block_0.getBuffer(), aggregate->getBuffer(),
          block_0.getLength(0), block_0.getLength(1), block_0.getLength(2)));
  EXPECT_EQ(block_0.getId()[0], aggregate->getId()[0]);
  EXPECT_EQ(block_0.getOffset(0), aggregate->getOffset(0));
  
  delete aggregate;
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

