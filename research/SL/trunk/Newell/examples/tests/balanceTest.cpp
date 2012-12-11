/*
 * File:   balanceTest.cpp
 * Author: den4gr
 *
 * Created on Nov 13, 2012, 4:03 PM
 */

#include <stdlib.h>
#include <iostream>
#include "../Balancer.h"
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

  TEST(BalanceTest, Strongest) {
    // initialize test
    const int kNumNodes(3); // number of cpu nodes
    const int kNumChildren(kNumNodes - 1); // number of gpus per cpu node
    const int kNumBlocks(100); // total number of blocks in the decomposition

    // initialize fake cluster
    Cluster cluster(kNumNodes);
    const double kHighWeight = 1.0;
    for (int i = 0; i < kNumNodes; ++i) {
      Node& node = cluster.getNode(i);
      node.setNumChildren(kNumChildren);
      if (0 == i) {
        node.setWeight(kHighWeight);
      } else {
        Node& gpu = node.getChild(i - 1);
        gpu.setWeight(kHighWeight);
      }
    }
    // make cpu node the strongest
    // initialize fake decomposition
    Decomposition decomp;
    for (int j = 0; j < kNumBlocks; ++j)
      decomp.addSubDomain(new SubDomain());

    // test components
    Balancer balancer;
    balancer.perfBalanceStrongestDevice(cluster, decomp);
    
    // check results
    for (unsigned int k = 0; k < cluster.getNumNodes(); ++k) {
      Node &node = cluster.getNode(k);
      if (0 == k) {
        // check cpu with work
        EXPECT_LT(0, node.getBalCount());

        // first node all children should have zero work
        for (unsigned int l = 0; l < node.getNumChildren(); ++l) {
          Node &gpu = node.getChild(l);
          EXPECT_EQ(0, gpu.getBalCount());
        }
      } else {
        // cpu nodes with rank > 0 should never have work assigned
        EXPECT_EQ(0, node.getBalCount());

        // check all gpus before gpu with work
        for (unsigned int m = 0; m < k - 1; ++m) {
          Node &gpu = node.getChild(m);
          EXPECT_EQ(0, gpu.getBalCount());
        }

        // check gpu with work
        EXPECT_LT(0, node.getChild(k - 1).getBalCount());

        // check all gpus after gpu with work
        for (unsigned int n = k; n < node.getNumChildren(); ++n) {
          Node &gpu = node.getChild(n);
          EXPECT_EQ(0, gpu.getBalCount());
        }
      }
    }
  }
} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

