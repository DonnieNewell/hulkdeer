/* -*- Mode: C ; indent-tabs-mode: nil ; c-file-style: "stroustrup" -*-

    CS 6620 - Compilers
    Stencil App Language Project
    Authors: Greg Faust, Sal Valente, derived from code by Jiayuan Meng


    File:   pathfinder-main.cpp     Contains a main routine to drive the pathfinder example.
*/

#include <stdlib.h>
#include <stdio.h>
#ifndef WIN32
#include <sys/time.h>
#else
#include<time.h>
#endif
#include "./cell.h"
#include "../Model.h"


/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

int J, K, L;
int* data;
int** space2D;
int*** space3D;
#define M_SEED 9
int pyramid_height;
int timesteps;

// #define BENCH_PRINT

double secondsElapsed(struct timeval start, struct timeval stop);

void init(int argc, char** argv) {
	if (argc == 6) {
		J = atoi(argv[1]);
		K = atoi(argv[2]);
                L = atoi(argv[3]);
                timesteps = atoi(argv[4]);
                pyramid_height=atoi(argv[5]);
	} else {
                printf("Usage: cell dim3 dim2 dim1 timesteps pyramid_height\n");
                exit(0);
        }
	data = new int[J * K * L];
        space2D = new int*[J * K];
	space3D = new int**[J];
	for (int n = 0; n < J * K; n++)
          space2D[n] = data + L * n;
	for(int n = 0; n < J; n++)
          space3D[n] = space2D + K * n;

	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < J * K * L; i++)
            data[i] = rand() % 2;
}

void printResults(int* data, int J, int K, int L) {
    int total = J * K * L;
    for (int n = 0; n < total; n++) {
        printf("%d ", data[n]);
    }
    printf("\n");
}

int bornMin = 5, bornMax = 8;
int dieMax = 3, dieMin = 10;

int main(int argc, char** argv) {
    init(argc, argv);
    struct timeval start, end;
    double total_sec = 0.0;
    const int kDevice = 0;
    gettimeofday(&start, NULL);
    runCell(data, J, K, L, timesteps, pyramid_height, bornMin, bornMax, dieMin,
            dieMax, kDevice);
    gettimeofday(&end, NULL);

    total_sec = secondsElapsed(start, end);
    printf("cudaCell time elapsed: %f\n", total_sec);

    delete [] data;
    delete [] space2D;
    delete [] space3D;

    return 0;
}

double secondsElapsed(struct timeval start, struct timeval stop) {
  return static_cast<double> ((stop.tv_sec - start.tv_sec) +
          (stop.tv_usec - start.tv_usec) / 1000000.0);
}
