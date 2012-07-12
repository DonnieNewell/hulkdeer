/* copyright 2012 Donnie Newell */
/* test numerical correctness of OpenMP and CUDA hotspot version */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../ompHotspot.h"
#include "../hotspot.h"

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
  const int kDataSize = 8;
  int iterations = 1;
  int device = 0;
  bool testPass;
  int grid_rows, grid_cols;
  float *ompMatrixTemp = NULL, *cudaMatrixTemp = NULL, *MatrixPower = NULL;
  char tfile[] = "../temp.dat";
  char pfile[] = "../power.dat";
  grid_rows = grid_cols = 2000;
  iterations = 1;

  // Read the power grid, which is read-only.
  int num_elements = grid_rows * grid_cols;
  MatrixPower = (float *) malloc(num_elements * sizeof (float));
  readInput(MatrixPower, grid_rows, grid_cols, pfile);

  // Read the temperature grid, which will change over time.
  ompMatrixTemp = (float *) malloc(num_elements * sizeof (float));
  cudaMatrixTemp = (float *) malloc(num_elements * sizeof (float));
  readInput(ompMatrixTemp, grid_rows, grid_cols, tfile);
  memcpy(cudaMatrixTemp, ompMatrixTemp, num_elements * sizeof (float));
  // printf("about to set the Data.\n");
  runOMPHotspotSetData(MatrixPower, num_elements);
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

  printf("starting correctness test.\n");
  printf("running OpenMP version.\n");

  runOMPHotspot(ompMatrixTemp, kDataSize, kDataSize, iterations, step_div_Cap,
          Rx, Ry, Rz);
  runOMPHotspotCleanup();

  printf("running CUDA version.\n");
  runHotspotSetData(MatrixPower, num_elements);

  printf("CUDA DATA PRE KERNEL ======================\n");
  printData(cudaMatrixTemp, kDataSize, kDataSize);
  runHotspot(cudaMatrixTemp, kDataSize, kDataSize, iterations, step_div_Cap,
          Rx, Ry, Rz, device);
  runHotspotCleanup();

  printf("ending correctness test.\n");
  testPass = compare(cudaMatrixTemp, ompMatrixTemp, kDataSize * kDataSize);
  printf("CUDA DATA ======================\n");
  printData(cudaMatrixTemp, kDataSize, kDataSize);
  printf("OpenMP DATA ======================\n");
  printData(ompMatrixTemp, kDataSize, kDataSize);

  free(ompMatrixTemp);
  ompMatrixTemp = NULL;
  free(cudaMatrixTemp);
  cudaMatrixTemp = NULL;
  free(MatrixPower);
  MatrixPower = NULL;

  if (testPass) {
    printf("Correctness passed\n");
    return 0;
  } else {
    printf("Correctness failed\n");
    return 1;
  }
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

DTYPE* initInput(const int kI, const int kJ) {
  DTYPE* data = new DTYPE[kI * kJ]();
  for (int i = 0; i < kI; ++i) {
    for (int j = 0; j < kJ; ++j) {
      int uidx = i * kJ + j;
      data[uidx] = (i + j) % 2;
    }
  }
  return data;
}

void printData(const DTYPE* data, const int I, const int J) {
  for (int i = I - 1; i >= 0; --i) {
    printf("[%d] ", i);
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
