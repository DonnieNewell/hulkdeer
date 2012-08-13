/* Copyright 2012 University of Virginia
  Author: Donnie Newell
*/

#include "../Profiler.h"
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  Profiler profiler;
  profiler.getNetworkLatency();

  MPI_Finalize();
}
