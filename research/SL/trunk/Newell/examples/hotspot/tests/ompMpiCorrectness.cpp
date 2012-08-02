/* copyright 2012 Donnie Newell */
/* test numerical correctness of distributed MPI cell version */

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include "../ompHotspot.h"
#include "../distributedHotspot.h"

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

// Forward declaration
DTYPE* initInput(const int kI, const int kJ);
void printData(const DTYPE* data, const int I, const int J);
bool compare(DTYPE* data1, DTYPE* data2, int length);
void readInput(float *vect, int grid_rows, int grid_cols, char *filename);
void writeOutput(float *vect, int grid_rows, int grid_cols, char *filename);

int main(int argc, char** argv) {
  const int kDataSize = 32;
  int iterations = 2;
  bool testPass;
  int returnCode, numTasks, rank;
  int grid_rows, grid_cols;
  float *ompMatrixTemp = NULL, *mpiMatrixTemp = NULL, *MatrixPower = NULL;
  char tfile[] = "../temp.dat";
  char pfile[] = "../power.dat";
  grid_rows = grid_cols = 2000;
  iterations = 1;
  int num_elements = grid_rows * grid_cols;
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

  returnCode = MPI_Init(&argc, &argv);
  if (returnCode != MPI_SUCCESS) {
    fprintf(stderr, "Error initializing MPI.\n");
    MPI_Abort(MPI_COMM_WORLD, returnCode);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("starting correctness test.\n");

  MatrixPower = new float[num_elements]();

  if (0 == rank) {
    // Read the power grid, which is read-only.
    readInput(MatrixPower, grid_rows, grid_cols, pfile);

    // Read the temperature grid, which will change over time.
    ompMatrixTemp = new float[num_elements]();
    mpiMatrixTemp = new float[num_elements]();
    readInput(ompMatrixTemp, grid_rows, grid_cols, tfile);
    memcpy(mpiMatrixTemp, ompMatrixTemp, num_elements * sizeof (float));
    // printf("about to set the Data.\n");
    runOMPHotspotSetData(MatrixPower, num_elements);
    printf("Set the OpenMP Data.\n");
    printf("running OpenMP version.\n");
    runOMPHotspot(ompMatrixTemp, kDataSize, kDataSize, iterations,
            step_div_Cap, Rx, Ry, Rz);
    runOMPHotspotCleanup();
    printf("running MPI version.\n");
  }
  runDistributedHotspotSetData(MatrixPower, num_elements);
  runDistributedHotspot(rank, numTasks, mpiMatrixTemp, kDataSize, kDataSize,
          iterations, step_div_Cap, Rx, Ry, Rz);
  returnCode = 1;
  if (0 == rank) {
    printf("mpiData:%p, ompData:%p\n", mpiMatrixTemp, ompMatrixTemp);
    testPass = compare(mpiMatrixTemp, ompMatrixTemp, kDataSize * kDataSize);
    printf("MPI DATA ======================\n");
    printData(mpiMatrixTemp, kDataSize, kDataSize);
    printf("OpenMP DATA ======================\n");
    printData(ompMatrixTemp, kDataSize, kDataSize);
    if (testPass) {
      printf("Correctness passed\n");
      returnCode = 0;
    } else {
      printf("Correctness failed\n");

    }
  }
  delete [] ompMatrixTemp;
  ompMatrixTemp = NULL;
  delete [] mpiMatrixTemp;
  mpiMatrixTemp = NULL;
  delete [] MatrixPower;
  MatrixPower = NULL;
  MPI_Finalize();
  return returnCode;
}

void readInput(float *vect, int grid_rows, int grid_cols, char *filename) {
  FILE *fp;
  int i, j;
  char str[80];

  printf("Reading %s...\n", filename);
  fp = fopen(filename, "r");
  if (fp == NULL) {
    perror(filename);
    return;
  }
  for (i = 0; i < grid_rows; i++) {
    for (j = 0; j < grid_cols; j++) {
      fgets(str, sizeof (str), fp);
      vect[i * grid_cols + j] = strtod(str, NULL);
    }
  }
  fclose(fp);
}

void writeOutput(float *vect, int grid_rows, int grid_cols, char *filename) {
  FILE *fp;
  int i, j, val, count, next;

  printf("Writing %s...\n", filename);
  fp = fopen(filename, "w");
  if (fp == NULL) {
    perror(filename);
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
          fprintf(fp, "%d ", val);
        else
          fprintf(fp, "%d(%d) ", val, count);
        val = next;
        count = 1;
      }
    }
    if (count == 1)
      fprintf(fp, "%d\n", val);
    else
      fprintf(fp, "%d(%d)\n", val, count);
  }
  fclose(fp);
}

DTYPE* initInput(int i, int j) {
  DTYPE* data = new DTYPE[i * j]();
  for (int x = 0; x < i * j; ++x) {
    data[x] = 1;
  }
  return data;
}

void printData(const DTYPE* data, const int I, const int J) {
  for (int i = I; i >= 0; --i) {
    printf("[%d]:\t", i);
    for (int j = 0; j < J; ++j) {
      printf("%f ", data[i * J + j]);
    }
    printf("\n");
  }
}

bool compare(DTYPE* data1, DTYPE* data2, int length) {
  for (int i = 0; i < length; ++i) {
    if (data1[i] != data2[i]) {
      printf("data1[%d]:%f != data2[%d]:%f\n", i, data1[i], i, data2[i]);
      return false;
    }
  }
  return true;
}