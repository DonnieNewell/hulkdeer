// -*- Mode: C++ ; c-file-style:"stroustrup"; indent-tabs-mode:nil; -*-

#include <stdio.h>
#include <stdlib.h>
#include <boost/scoped_array.hpp>
#ifndef WIN32
#include <sys/time.h>
#else
#include<time.h>
#endif
#include "./distributedHotspot.h"
#include "../comm.h"
#include "mpi.h"

/* maximum power density possible (say 300W for a 10mm x 10mm chip) */
#define MAX_PD (3.0e6)
/* required precision in degrees */
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor */
#define FACTOR_CHIP 0.5

/* chip parameters */
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
int pyramid_height = 2;
bool perform_load_balancing = false;
int device_configuration = 0;

// Forward declaration
void readInput(float *vect, int grid_rows, int grid_cols, const char *kFilename);
void writeOutput(float *vect, int grid_rows, int grid_cols, const char *kFilename);

int main(int argc, char** argv) {
  int return_code = 0, num_tasks = 0, my_rank = 0;
  return_code = MPI_Init(&argc, &argv);
  if (return_code != MPI_SUCCESS) {
    fprintf(stderr, "Error initializing MPI.\n");
    MPI_Abort(MPI_COMM_WORLD, return_code);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int grid_rows = 0, grid_cols = 0, iterations = 0;
  boost::scoped_array<float> matrix_temp;
  boost::scoped_array<float> matrix_power;
  const char kTempFile[] = "temp.dat";
  const char kPowerFile[] = "power.dat";
  const char kOutFile[] = "output.dat";
  const int kRootIndex = 0;
  int num_blocks_per_dimension = 0;
  if (argc >= 3) {
    grid_rows = atoi(argv[1]);
    grid_cols = atoi(argv[1]);
    iterations = atoi(argv[2]);
    if (argc >= 4)
      setenv("BLOCKSIZE", argv[3], 1);
    if (argc >= 5) {
      setenv("HEIGHT", argv[4], 1);
      pyramid_height = atoi(argv[4]);
    }
    if (argc >= 6)
      num_blocks_per_dimension = atoi(argv[5]);
    if (argc >= 7)
      perform_load_balancing = (atoi(argv[6]) != 0) ? true : false;
    if (argc >= 8)
      device_configuration = atoi(argv[7]);
  } else {
    printf("Usage: hotspot grid_rows_and_cols iterations [blocksize] [pyramid_height] [blocks_per_dimension] [load_balance] [device_config]\n");
    return 0;
  }

  // Read the power grid, which is read-only.
  int num_elements = grid_rows * grid_cols;
  matrix_power.reset(new float[num_elements]());
  if (my_rank == kRootIndex) {
    readInput(matrix_power.get(), grid_rows, grid_cols, kPowerFile);

    // Read the temperature grid, which will change over time.
    matrix_temp.reset(new float[num_elements]());
    readInput(matrix_temp.get(), grid_rows, grid_cols, kTempFile);
  }
  // printf("about to set the Data.\n");
  runDistributedHotspotSetData(matrix_power.get(), num_elements);
  printf("Set the Data.\n");

  float grid_width = chip_width / grid_cols;
  float grid_height = chip_height / grid_rows;
  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  // TODO: Invert Rx, Ry, Rz?
  float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);
  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;
  float step_div_Cap = step / Cap;

  struct timeval starttime, endtime;
  double sec;

  gettimeofday(&starttime, NULL);
  runDistributedHotspot(my_rank, num_tasks, matrix_temp.get(), grid_cols,
          grid_rows, iterations, pyramid_height, step_div_Cap, Rx, Ry, Rz,
          num_blocks_per_dimension, perform_load_balancing,
          device_configuration);
  gettimeofday(&endtime, NULL);

  sec = secondsElapsed(starttime, endtime);
  if (0 == my_rank) {
    printf("Total time=%f\n", sec);
    writeOutput(matrix_temp.get(), grid_rows, grid_cols, kOutFile);
  }

  MPI_Finalize();
  return 0;
}

void readInput(float *vect, int grid_rows, int grid_cols, const char *kFilename) {
  FILE *fp;
  int i, j;
  char str[80];

  printf("Reading %s...\n", kFilename);
  fp = fopen(kFilename, "r");
  if (fp == NULL) {
    perror(kFilename);
    return;
  }
  for (i = 0; i < grid_rows; i++) {
    for (j = 0; j < grid_cols; j++) {
      if (NULL != fgets(str, sizeof (str), fp))
        vect[i * grid_cols + j] = strtod(str, NULL);
    }
  }
  fclose(fp);
}

void writeOutput(float *vect, int grid_rows, int grid_cols, const char *kFilename) {
  FILE *file_pointer;
  int i, j, val, count, next;

  printf("Writing %s...\n", kFilename);
  file_pointer = fopen(kFilename, "w");
  if (file_pointer == NULL) {
    perror(kFilename);
    return;
  }
  for (i = 0; i < grid_rows; i++) {
    j = 0;
    val = (int) vect[i * grid_cols + j];
    count = 1;
    for (j = 1; j < grid_cols; j++) {
      next = (int) vect[i * grid_cols + j];
      if (next == val) {
        count++;
      } else {
        if (count == 1)
          fprintf(file_pointer, "%d ", val);
        else
          fprintf(file_pointer, "%d(%d) ", val, count);
        val = next;
        count = 1;
      }
    }
    if (count == 1)
      fprintf(file_pointer, "%d\n", val);
    else
      fprintf(file_pointer, "%d(%d)\n", val, count);
  }
  fclose(file_pointer);
}
