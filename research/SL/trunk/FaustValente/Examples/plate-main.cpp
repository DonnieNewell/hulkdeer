// -*- Mode: C++ ; c-file-style:"stroustrup"; indent-tabs-mode:nil; -*-

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "plate.h"
#include "Model.h"

/**
 * Call the runPlate() function that was generated from the
 * stencil language description of the heated plate problem.
 */
int main(int argc, char **argv)
{
    int input_width, input_height, timesteps, x, y;
    struct timeval starttime, endtime;
    float *host_data;
    long usec;

    input_width = 40;
    input_height = 40;
    timesteps = 100;
    if (argc >= 3)
    {
        input_width = atoi(argv[1]);
        input_height = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        timesteps = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        // Pyramid height
        setenv("HEIGHT", argv[4], 1);
    }
    host_data = (float *) calloc(input_width * input_height, sizeof(float));

#ifdef STATISTICS
    // Set iteration count so that kernel is called at least 30 times.
    // The maximum pyramid height is 10, so iterations = 300.
    for (int i=500; i<=input_width; i+= 500) runPlate(host_data, i, i, 300);
#else
    gettimeofday(&starttime, NULL);
    SL_MPI_Init();
    runPlate(host_data, input_width, input_height, timesteps);
    SL_MPI_Finalize();
    gettimeofday(&endtime, NULL);
    usec = ((endtime.tv_sec - starttime.tv_sec) * 1000000 +
            (endtime.tv_usec - starttime.tv_usec));
    printf("time=%ld\n", usec);

    // Output final hotplate for debugging.
    const char *rep = " .:;|%O0&8#";
    for (y = 0; y < input_height; y++)
    {
        for (x = 0; x < input_width; x++)
            printf("%c", rep[(int)(host_data[y*input_width+x] / 10)]);
        printf("\n");
    }
#endif

    printCudaDeviceProps(getCurrentCudaDeviceProps());

    return 0;
}
