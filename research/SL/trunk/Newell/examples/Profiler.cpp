/* copyright 2012 University of Virginia
  Profiler.cpp
  author: Donnie Newell (den4gr@virginia.edu)
 */

#include "Profiler.h"
#include "mpi.h"
#include <math.h>
#include <boost/scoped_array.hpp>

Profiler::Profiler() {
}

void Profiler::getNetworkLatency() {
  int my_rank = -1, number_tasks = -1;
  const int kRootRank = 0;
  const double kBase = 2.0;
  const double kInitialPower = 5.0;
  const double kPowerLimit = 11.0;
  const int kSendAndReceive = 2;
  const int kNumberIterations = 20;
  const int kMaxElements = static_cast<int> (pow(kBase, kPowerLimit));
  MPI_Status status;
  const int kTag = 1;
  double end_time = 0.0, start_time = 0.0;
  int buffer[kMaxElements];
  void* mpi_pointer = static_cast<void*> (buffer);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &number_tasks);
  printf("number_tasks:%d my_rank:%d kMaxElements:%d \n", number_tasks, my_rank,
          kMaxElements);
  const int kNumberTimes = static_cast<int>(kPowerLimit - kInitialPower);
  boost::scoped_array<double> times(new double[kNumberTimes]());
  
  //load array into cache
  for (int i = 0; i < kMaxElements; ++i) buffer[i] = i;
  
  if (kRootRank == my_rank) {
    for (double power = kInitialPower; power < kPowerLimit; ++power) {
      start_time = MPI_Wtime();
      for (int node_rank = 0; node_rank < number_tasks; ++node_rank) {
        if (kRootRank == node_rank) continue;
        
        int number_elements = static_cast<int> (pow(kBase, power));
        for (int iteration = 0; iteration < kNumberIterations; ++iteration) {
          MPI_Send(mpi_pointer, number_elements, MPI_INT, node_rank, kTag,
                  MPI_COMM_WORLD);
          MPI_Recv(mpi_pointer, number_elements, MPI_INT, node_rank, kTag,
                  MPI_COMM_WORLD, &status);
        }
      }
      end_time = MPI_Wtime();
      int index = static_cast<int>(power - kInitialPower);
      times[index] = (end_time - start_time) /
                     (number_tasks - 1) * kNumberIterations * kSendAndReceive;
    }
    
    for (int i = 0; i < kPowerLimit - kInitialPower; ++i) {
      printf("buffer size(%f ^ %f) latency: %f\n", kBase, i + kInitialPower, 
              times[i]);
    }
  } else {
    for (double power = kInitialPower; power < kPowerLimit; ++power) {
      int number_elements = static_cast<int> (pow(kBase, power));
      for (int iteration = 0; iteration < kNumberIterations; ++iteration) {
        MPI_Recv(mpi_pointer, number_elements, MPI_INT, kRootRank, kTag,
                MPI_COMM_WORLD, &status);
        MPI_Send(mpi_pointer, number_elements, MPI_INT, kRootRank, kTag,
                MPI_COMM_WORLD);
      }
    }
  }
}

