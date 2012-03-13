/* -*- Mode: C ; indent-tabs-mode: nil ; c-file-style: "stroustrup" -*-

    CS 6620 - Compilers
    Stencil App Language Project
    Authors: Greg Faust, Sal Valente, derived from code by Jiayuan Meng
    

    File:   pathfinder-main.cpp     Contains a main routine to drive the pathfinder example.
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "1dmpi.h"
#include "Model.h"


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

void
init(int argc, char** argv)
{
	if(argc==6){
		J = atoi(argv[1]);
		K = atoi(argv[2]);
                L = atoi(argv[3]);
                timesteps = atoi(argv[4]);
                pyramid_height=atoi(argv[5]);
	}else{
                printf("Usage: cell dim3 dim2 dim1 timesteps pyramid_height\n");
                exit(0);
        }
	data = new int[J*K*L];
        space2D = new int*[J*K];
	space3D = new int**[J];
	for(int n=0; n<J*K; n++)
          space2D[n]=data+L*n;
	for(int n=0; n<J; n++)
          space3D[n]=space2D+K*n;

	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < J*K*L; i++)
            data[i] = rand()%2;

}

void printResults(int* data, int J, int K, int L)
{
    int total = J*K*L;
    for(int n = 0; n < total; n++){
        printf("%d ", data[n]);
    }
    printf("\n");
}

int bornMin = 5, bornMax = 8;
int dieMax = 3, dieMin = 10;

int main(int argc, char** argv)
{
    init(argc, argv);

    //SL_MPI_Init();
#ifdef STATISTICS
    for (int i=40; i<=J; i += 20)
    {
        // Set iteration count so that kernel is called at least 30 times.
        // The maximum pyramid height is 3, so iterations = 90.
        runCell(data, i, i, i, 90, bornMin, bornMax, dieMin, dieMax);
    }
#else
    runCell(data, J, K, L, timesteps, bornMin, bornMax, dieMin, dieMax);
    //SL_Finalize();

#ifdef BENCH_PRINT
    printResults(data, J, K, L);
#endif
#endif

    delete [] data;
    delete [] space2D;
    delete [] space3D;

    return 0;
}

