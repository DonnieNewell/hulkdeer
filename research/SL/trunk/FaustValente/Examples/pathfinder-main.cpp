/* -*- Mode: C ; indent-tabs-mode: nil ; c-file-style: "stroustrup" -*-

    CS 6620 - Compilers
    Stencil App Language Project
    Authors: Greg Faust, Sal Valente, derived from code by Jiayuan Meng
    

    File:   pathfinder-main.cpp     Contains a main routine to drive the pathfinder example.
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "pathfinder.h"
#include "Model.h"

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

// #define BENCH_PRINT

void
init(int argc, char** argv)
{
    if(argc==4)
    {
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
        pyramid_height=atoi(argv[3]);
    }
    else
    {
        printf("Usage: dynproc row_len col_len pyramid_height\n");
        exit(0);
    }
#ifdef STATISTICS
    // We would like to get 30 iterations.
    // But the data would be too big.
    // Try for 2!!
    // Nope, can only get 1!!
    rows = 256;
#endif
    data = new int[rows*cols];
    wall = new int*[rows];
    for(int n=0; n<rows; n++) wall[n]=data+cols*n;
    result = new int[cols];
    
    int seed = M_SEED;
    srand(seed);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
}

int main(int argc, char** argv)
{
    // int num_devices;
    // cudaGetDeviceCount(&num_devices);
    // if (num_devices > 1) cudaSetDevice(DEVICE);

    init(argc, argv);
#ifdef BENCH_PRINT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]) ;
        }
        printf("\n") ;
    }
#endif

    // The algorithm calculate over a row at a time.
    // Therefore, the size is the number of columns,
    // And the number of rows is the number of time intervals.


    SL_MPI_Init();
#ifdef STATISTICS
    for (int i=100000; i<=cols; i += 300000)
    {
        // Set iteration count so that kernel is called at least 30 times.
        // The maximum pyramid height is 255, so iterations = 7650.
        runPathFinderSetData(data, rows * cols);
        runPathFinder(result, i, rows);
    }
#else
    runPathFinderSetData(data, rows * cols);
    runPathFinder(result, cols, rows-1);
#endif
    SL_MPI_Finalize();

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
            printf("%d ",data[i]) ;
    printf("\n") ;
    for (int i = 0; i < cols; i++)
            printf("%d ",result[i]) ;
    printf("\n") ;
#endif

    delete []data;
    delete []wall;
    delete []result;

    return 0;
}
