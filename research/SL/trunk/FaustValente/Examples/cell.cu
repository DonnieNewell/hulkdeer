// -*- Mode: C++ ; c-file-style:"stroustrup"; indent-tabs-mode:nil; -*-

#include <stdio.h>
#include <stdlib.h>
#include "Model.cu"

#define DTYPE int

// The size of the tile is calculated at compile time by the SL processor.
// But the data array is statically sized.
// So, make these are big as they can get.
// Changed to be large enough for fermi
// (int)cube_rt(1024) = 10
#define TILE_WIDTH  10
#define TILE_HEIGHT 10
#define TILE_DEPTH  10

/**
 * Block of memory shared by threads working on a single tile.
 * Contains all necessary cell values and edge values from the
 * previous iteration.
 */
__shared__ DTYPE shmem[TILE_DEPTH][TILE_HEIGHT][TILE_WIDTH];

__device__ DTYPE get(int x, int y, int z)
{
    return shmem[threadIdx.z+z][threadIdx.y+y][threadIdx.x+x];
}

// Macro to read global read only data from within CellValue code.
#define read(offset)(ro_data[offset])

__device__ DTYPE CellValue(dim3 input_size, int x, int y, int z, DTYPE *ro_data
                           , int bornMin, int bornMax, int dieMin, int dieMax)
{
    int orig = get(0, 0, 0);
    int sum = 0;
    int i, j, k;
    for (i = -1; i <= 1; i++)
        for (j = -1; j <= 1; j++)
	    for (k = -1; k <= 1; k++)
	    	sum += get(i, j, k);
    sum -= orig;
    int retval;
    if(orig>0 && (sum <= dieMax || sum >= dieMin)) retval = 0;
    else if (orig==0 && (sum >= bornMin && sum <= bornMax)) retval = 1;
    else retval = orig;    
    return (retval);
}

/**
 * Each thread runs this kernel to calculate the value at one particular
 * cell in one particular iteration.
 */

// We need to declare it C style naming.
// This avoids name mangling and allows us to get attributes about the kernel call from Cuda.
// Its possible to do this with a C++ interface, but that will only run on certain devices.
// This technique is older and therefore more reliable across Cuda devices.
extern "C" {
void runCellKernel(dim3 input_size, dim3 stencil_size,
                          DTYPE *input, DTYPE *output, int pyramid_height,
                          DTYPE *ro_data
                          , int bornMin, int bornMax, int dieMin, int dieMax);
    }

__global__
void runCellKernel(dim3 input_size, dim3 stencil_size,
                          DTYPE *input, DTYPE *output, int pyramid_height,
                          DTYPE *ro_data
                          , int bornMin, int bornMax, int dieMin, int dieMax)
{
    dim3 border;
    int bx, by, bz, tx, ty, tz, x, y, z, ex, ey, ez, uidx, iter, inside;
    DTYPE value;

    // (bx, by, bz) is the location in the input of the top left of this block.
    border.x = pyramid_height * stencil_size.x;
    border.y = pyramid_height * stencil_size.y;
    border.z = pyramid_height * stencil_size.z;
    bx = blockIdx.x * (blockDim.x - 2*border.x) - border.x;
    // These changed by Greg Faust to fix the fact that
    //     grids in CUDA cannot have 3 dimensions.
    // This parallels the same fix Jiayuan Meng used in his code for this issue.
    // by = blockIdx.y * (blockDim.y - 2*border.y) - border.y;
    // bz = blockIdx.z * (blockDim.z - 2*border.z) - border.z;
    int BS = blockDim.x;
    by = (blockIdx.y/BS) * (BS - 2*border.y) - border.y;
    bz = (blockIdx.y%BS) * (BS - 2*border.z) - border.z;

    // (x, y, z) is the location in the input of this thread.
    tx = threadIdx.x;
    ty = threadIdx.y;
    tz = threadIdx.z;
    x = bx + tx;
    y = by + ty;
    z = bz + tz;

    // (ex, ey, ez) = (x, y, z) pushed into the boundaries of the input.
    ex = x;
    ey = y;
    ez = z;
    if (ex < 0) ex = 0;
    if (ey < 0) ey = 0;
    if (ez < 0) ez = 0;
    if (ex >= input_size.x) ex = input_size.x-1;
    if (ey >= input_size.y) ey = input_size.y-1;
    if (ez >= input_size.z) ez = input_size.z-1;

    // Get current cell value or edge value.
    // uidx = ez + input_size.y * (ey * input_size.x + ex);
    uidx = ex + input_size.x * (ey + ez * input_size.y);
    value = input[uidx];
    inside = ((x == ex) && (y == ey) && (z == ez));

    // Store value in shared memory for stencil calculations, and go.
    shmem[tz][ty][tx] = value;
    iter = 0;
    border.x = border.y = border.z = 0;
    while (true)
    {
        __syncthreads();
        iter++;
        if (inside)
        {
            border.x += stencil_size.x;
            border.y += stencil_size.y;
            border.z += stencil_size.z;
            inside = ((tx >= border.x) && (tx < blockDim.x-border.x) &&
                      (ty >= border.y) && (ty < blockDim.y-border.y) &&
                      (tz >= border.z) && (tz < blockDim.z-border.z));
        }
        if (inside)
        {
            value = CellValue(input_size, x, y, z, ro_data
                              , bornMin, bornMax, dieMin, dieMax);
        }
        if (iter >= pyramid_height)
        {
            if (inside)
                output[uidx] = value;
            break;
        }
        __syncthreads();

        shmem[tz][ty][tx] = value;
    }
}

/**
 * Store data between calls to SetData() and run().
 * This is basically a hack.
 */
static DTYPE *global_ro_data = NULL;

/**
 * Function exported to do the entire stencil computation.
 */
void runCell(DTYPE *host_data, int x_max, int y_max, int z_max, int iterations
                    , int bornMin, int bornMax, int dieMin, int dieMax)
{
    // User-specific parameters
    dim3 input_size(x_max, y_max, z_max);
    dim3 stencil_size(1,1,1);

    // Host to device
    DTYPE *device_input, *device_output;
    int num_bytes = input_size.x * input_size.y * input_size.z * sizeof(DTYPE);
    cudaMalloc((void **) &device_input, num_bytes);
    cudaMalloc((void **) &device_output, num_bytes);
    cudaMemcpy(device_input, host_data, num_bytes, cudaMemcpyHostToDevice);

#ifdef STATISTICS
    struct timeval trainingstarttime, trainingendtime;
    unsigned int trainingusec;
    gettimeofday(&trainingstarttime, NULL);                                       
#endif

    // Setup the structure that holds parameters for the application.
    // And from that, get the block size.
    char * KernelName = "runCellKernel";
    dim3 tile_size = initSAProps(3, input_size, stencil_size, iterations, sizeof(DTYPE), KernelName);

    dim3 border, tile_data_size, grid_dims;

    // Now ready for the training period.
    // Need to get some timings of small kernel runs.
    // TODO It would be faster if these could be 0 and 1 heights instead of 1 and 2.
    int pyramid_height = 2;
    filldim3(&border, pyramid_height * stencil_size.x, pyramid_height * stencil_size.y, pyramid_height * stencil_size.z);
    filldim3(&tile_data_size, tile_size.x - 2*border.x, tile_size.y - 2*border.y, tile_size.z - 2*border.z);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x), div_ceil(input_size.y, tile_data_size.y)*div_ceil(input_size.z, tile_data_size.z));
    unsigned int twoIterTime;
    timeInMicroSeconds(twoIterTime, (runCellKernel<<< grid_dims, tile_size >>>(
                                    input_size, stencil_size, device_input, device_output,
                                    pyramid_height, global_ro_data
                                    , bornMin, bornMax, dieMin, dieMax)));
    pyramid_height = 1;
    filldim3(&border, pyramid_height * stencil_size.x, pyramid_height * stencil_size.y, pyramid_height * stencil_size.z);
    filldim3(&tile_data_size, tile_size.x - 2*border.x, tile_size.y - 2*border.y, tile_size.z - 2*border.z);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x), div_ceil(input_size.y, tile_data_size.y)*div_ceil(input_size.z, tile_data_size.z));
    unsigned int oneIterTime;
    timeInMicroSeconds(oneIterTime, (runCellKernel<<< grid_dims, tile_size >>>(
                                    input_size, stencil_size, device_input, device_output,
                                    pyramid_height, global_ro_data
                                    , bornMin, bornMax, dieMin, dieMax)));

#ifdef STATISTICS
/////////////////////////////////////////////////////////////////////////////////////
// Start of code to gather statistics to hone model.  Remove in final version.
////////////////////////////////////////////////////////////////////////////////////

    fprintf(stderr, "***********************************Start of a new Run****************************************\n");
    fprintf(stderr, "Data Size=%d, Tile Size=%d Iteration Count=%d\n", input_size.x, tile_size.x, iterations);

    // Precalculate the pyramid height so we can get stats on the calculated value.
    int calcMinPyramid = calcPyramidHeight(grid_dims, oneIterTime, twoIterTime);

    gettimeofday(&trainingendtime, NULL);
    trainingusec = ((trainingendtime.tv_sec - trainingstarttime.tv_sec) * 1000000 +             
                    (trainingendtime.tv_usec - trainingstarttime.tv_usec));                       

    // Get second best for same reason.
    int secondMinPyramid = getSecond(calcMinPyramid);

    // Gather statistics to help hone model.
    double calcMinTime, secondMinTime;
    double actualMinTime = 1000000000;
    int actualMinPyramid;
    // Now let's just try them all to see what the optimal pyramid height is.
    for (int i=1; i<tile_size.x/(2 * stencil_size.x); i++)
    {
        int pyramid_height = i;

        // Now we can calculate the other sizes.
        dim3 border(pyramid_height * stencil_size.x,
                    pyramid_height * stencil_size.y,
                    pyramid_height * stencil_size.z);
        dim3 tile_data_size(tile_size.x - 2*border.x,
                            tile_size.y - 2*border.y,
                            tile_size.z - 2*border.z);
        dim3 grid_dims(div_ceil(input_size.x, tile_data_size.x),
                       div_ceil(input_size.y, tile_data_size.y)*
                       div_ceil(input_size.z, tile_data_size.z));

        uint32_t time;
        timeInMicroSeconds(time, (runCellKernel<<< grid_dims, tile_size >>>(
                                      input_size, stencil_size, device_input, device_output,
                                      i, global_ro_data
                                      , bornMin, bornMax, dieMin, dieMax)));
        
        double timePer = ((double)time)/i;
        if (i == calcMinPyramid) calcMinTime = timePer;
        if (i == secondMinPyramid) secondMinTime = timePer;
        if (timePer < actualMinTime)
        {
            actualMinPyramid = i;
            actualMinTime = timePer;
        }
        // fprintf(stderr, "Pyramid Height=%d, time=%u, Time per iteration=%f.\n", i, time, ((double)time/i));
    }

    // Now we can output some statistics.
    double firstError = ((1. - (actualMinTime/calcMinTime)) * 100.);
    double secondError = ((1. - (actualMinTime/secondMinTime)) * 100.);
    fprintf(stderr, "Size %d BestHeight %d CalcHeight %d %%Slowdown %4.2f CalcSecond %d %%Slowdown %4.2f MinSlowdown %4.2f\n", 
            input_size.x, actualMinPyramid, calcMinPyramid, firstError, secondMinPyramid, secondError, MIN(firstError, secondError));

/////////////////////////////////////////////////////////////////////////////////////
// End of code to gather statistics to hone model.  Remove in final version.
////////////////////////////////////////////////////////////////////////////////////
#endif

#ifdef STATISTICS

    for (int i=1; i<tile_size.x/(2 * stencil_size.x); i++)
    {
        struct timeval starttime, endtime;
        unsigned int usec2;
        gettimeofday(&starttime, NULL);                                       

        pyramid_height=i;

#else
    // Now we can calculate the pyramid height.
    pyramid_height = calcPyramidHeight(grid_dims, oneIterTime, twoIterTime);
#endif

    // And use the result to calculate various sizes.
    filldim3(&border, pyramid_height * stencil_size.x, pyramid_height * stencil_size.y, pyramid_height * stencil_size.z);
    filldim3(&tile_data_size, tile_size.x - 2*border.x, tile_size.y - 2*border.y, tile_size.z - 2*border.z);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x), div_ceil(input_size.y, tile_data_size.y)*div_ceil(input_size.z, tile_data_size.z));

    // Run computation
    for (int iter = 0; iter < iterations; iter += pyramid_height)
    {
        if (iter + pyramid_height > iterations)
            pyramid_height = iterations - iter;
        runCellKernel<<< grid_dims, tile_size >>>(
            input_size, stencil_size, device_input, device_output,
            pyramid_height, global_ro_data
            , bornMin, bornMax, dieMin, dieMax);
        DTYPE *temp = device_input;
        device_input = device_output;
        device_output = temp;
    }

#ifdef STATISTICS
    // Synch the threads to make sure everything is done before taking a timing.
    CUDA_SAFE_THREAD_SYNC();
    gettimeofday(&endtime, NULL);                                       
    usec2 = ((endtime.tv_sec - starttime.tv_sec) * 1000000 +             
             (endtime.tv_usec - starttime.tv_usec));                       
    fprintf(stderr, "Actual pyramid=%d, Actual iteration time=%Lu, Actual Total time=%lu\n", i, usec2, usec2+trainingusec);
    }
#endif

    // Device to host
    cudaMemcpy(host_data, device_input, num_bytes, cudaMemcpyDeviceToHost);
    cudaFree(device_input);
    cudaFree(device_output);
    if (global_ro_data != NULL)
    {
        cudaFree(global_ro_data);
        global_ro_data = NULL;
    }
}

/**
 * Store unnamed data on device.
 */
void runCellSetData(DTYPE *host_data, int num_elements)
{
    int num_bytes = sizeof(DTYPE) * num_elements;
    cudaMalloc((void **) &global_ro_data, num_bytes);
    cudaMemcpy(global_ro_data, host_data, num_bytes, cudaMemcpyHostToDevice);
}
