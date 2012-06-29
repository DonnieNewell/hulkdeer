// -*- Mode: C++ ; c-file-style:"stroustrup"; indent-tabs-mode:nil; -*-

#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <sys/time.h>
#else
#include<time.h>
#endif
#include "distributedHotspot.h"
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

// Forward declaration
void readInput(float *vect, int grid_rows, int grid_cols, char *filename);
void writeOutput(float *vect, int grid_rows, int grid_cols, char *filename);

int main(int argc, char** argv) {
    int rc, numTasks, rank;
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "Error initializing MPI.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int grid_rows, grid_cols, iterations;
    float *MatrixTemp = NULL, *MatrixPower = NULL;
    char tfile[] = "temp.dat";
    char pfile[] = "power.dat";
    char ofile[] = "output.dat";
    const int kRootIndex = 0;

    if (argc >= 3) {
        grid_rows = atoi(argv[1]);
        grid_cols = atoi(argv[1]);
        iterations = atoi(argv[2]);
        if (argc >= 4) setenv("BLOCKSIZE", argv[3], 1);
        if (argc >= 5) setenv("HEIGHT", argv[4], 1);
    } else {
        printf("Usage: hotspot grid_rows_and_cols iterations [blocksize]\n");
        return 0;
    }

    // Read the power grid, which is read-only.
    int num_elements = grid_rows * grid_cols;
    MatrixPower = (float *) malloc(num_elements * sizeof (float));
    if (rank == kRootIndex) {
        readInput(MatrixPower, grid_rows, grid_cols, pfile);

        // Read the temperature grid, which will change over time.
        MatrixTemp = (float *) malloc(num_elements * sizeof (float));
        readInput(MatrixTemp, grid_rows, grid_cols, tfile);
    }
    runDistributedHotspotSetData(MatrixPower, num_elements);

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
    long usec;


    gettimeofday(&starttime, NULL);
    runDistributedHotspot(rank, numTasks, MatrixTemp, grid_cols, grid_rows,
            iterations, step_div_Cap, Rx, Ry, Rz);
    gettimeofday(&endtime, NULL);

    usec = ((endtime.tv_sec - starttime.tv_sec) * 1000000 +
            (endtime.tv_usec - starttime.tv_usec));
    printf("Total time=%ld\n", usec);
    writeOutput(MatrixTemp, grid_rows, grid_cols, ofile);

    free(MatrixTemp);
    free(MatrixPower);
    return 0;
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
            if (next == val)
                count++;
            else {
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
