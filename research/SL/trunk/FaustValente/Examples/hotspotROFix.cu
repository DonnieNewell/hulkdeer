// -*- Mode: C++ ; c-file-style:"stroustrup"; indent-tabs-mode:nil; -*-

#include <stdio.h>
#include <stdlib.h>
#include "Model.cu"

#define DTYPE float

// The size of the tile is calculated at compile time by the SL processor.
// But the data array is statically sized.
// So, make these are big as they can get.
// (int)sqrt(512) = 22
#define TILE_WIDTH  32
#define TILE_HEIGHT 32

/**
 * Block of memory shared by threads working on a single tile.
 * Contains all necessary cell values and edge values from the
 * previous iteration.
 */
__shared__ DTYPE shmem[TILE_HEIGHT][TILE_WIDTH];

__device__ DTYPE get(int x, int y)
{
    return shmem[threadIdx.y+y][threadIdx.x+x];
}

/**
 * Store data between calls to SetData() and run().
 * This is basically a hack.
 */
static DTYPE *global_ro_data = NULL;

// Macro to read global read only data from within CellValue code.
#define read(offset)(ro_data[offset])

__device__ DTYPE CellValue(dim3 input_size, int x, int y, DTYPE pvalue //*ro_data
                           , float step_div_Cap, float Rx, float Ry, float Rz)
{
    float value, term1, term2, term3, sum;
    // pvalue = read(y * input_size.x + x);
    value = get(0, 0);
    term1 = (get(0, 1) + get(0, -1) - value - value) / Ry;
    term2 = (get(1, 0) + get(-1, 0) - value - value) / Rx;
    term3 = (80.0 - value) / Rz;
    sum = pvalue + term1 + term2 + term3;
    return(value + step_div_Cap * sum);
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
void runHotspotKernel(dim3 input_size, dim3 stencil_size,
                          DTYPE *input, DTYPE *output, int pyramid_height,
                          DTYPE *ro_data
                          , float step_div_Cap, float Rx, float Ry, float Rz);
    }

__global__
void runHotspotKernel(dim3 input_size, dim3 stencil_size,
                          DTYPE *input, DTYPE *output, int pyramid_height,
                          DTYPE *ro_data
                          , float step_div_Cap, float Rx, float Ry, float Rz)
{
    dim3 border;
    int bx, by, tx, ty, x, y, ex, ey, uidx, iter, inside;
    DTYPE value;

    // (bx, by) is the location in the input of the top left of this block.
    border.x = pyramid_height * stencil_size.x;
    border.y = pyramid_height * stencil_size.y;
    bx = blockIdx.x * (blockDim.x - 2*border.x) - border.x;
    by = blockIdx.y * (blockDim.y - 2*border.y) - border.y;
    // (x, y) is the location in the input of this thread.
    tx = threadIdx.x;
    ty = threadIdx.y;
    x = bx + tx;
    y = by + ty;

    // (ex, ey) = (x, y) pushed into the boundaries of the input.
    ex = x;
    ey = y;
    if (ex < 0) ex = 0;
    if (ey < 0) ey = 0;
    if (ex >= input_size.x) ex = input_size.x-1;
    if (ey >= input_size.y) ey = input_size.y-1;

    // Get current cell value or edge value.
    uidx = ey * input_size.x + ex;
    value = input[uidx];
    inside = ((x == ex) && (y == ey));

    // Store value in shared memory for stencil calculations, and go.
    DTYPE pvalue = 0;
    // This works worse for PH=1, but better otherwise.
    if (inside) pvalue = ro_data[y * input_size.x + x];
    shmem[ty][tx] = value;
    iter = 0;
    border.x = border.y = 0;
    while (true)
    {
        __syncthreads();
        iter++;
        if (inside)
        {
            border.x += stencil_size.x;
            border.y += stencil_size.y;
            inside = ((tx >= border.x) && (tx < blockDim.x-border.x) &&
                      (ty >= border.y) && (ty < blockDim.y-border.y));
        }
        if (inside)
        {
            // This works better for PH=1, but worse otherwise.
            // if (iter == 1) pvalue = ro_data[y * input_size.x + x];
            value = CellValue(input_size, x, y, pvalue
                              , step_div_Cap, Rx, Ry, Rz);
        }
        if (iter >= pyramid_height)
        {
            if (inside)
                output[uidx] = value;
            break;
        }
        __syncthreads();
        shmem[ty][tx] = value;
    }
}

/**
 * Function exported to do the entire stencil computation.
 */
void runHotspot(DTYPE *host_data, int x_max, int y_max, int iterations
                    , float step_div_Cap, float Rx, float Ry, float Rz)
{
    // User-specific parameters
    dim3 input_size(x_max, y_max);
    dim3 stencil_size(1,1);

    // Host to device
    DTYPE *device_input, *device_output;
    int num_bytes = input_size.x * input_size.y * sizeof(DTYPE);
    cudaMalloc((void **) &device_input, num_bytes);
    cudaMalloc((void **) &device_output, num_bytes);
    cudaMemcpy(device_input, host_data, num_bytes, cudaMemcpyHostToDevice);

    // Setup the structure that holds parameters for the application.
    // And from that, get the block size.
    char * KernelName = "runHotspotKernel";
    dim3 tile_size = initSAProps(2, input_size, stencil_size, iterations, sizeof(DTYPE), KernelName);

    dim3 border, tile_data_size, grid_dims;

    // Now ready for the training period.
    // Need to get some timings of small kernel runs.
    // TODO It would be faster if these could be 0 and 1 heights instead of 1 and 2.
    int pyramid_height = 2;
    filldim3(&border, pyramid_height * stencil_size.x, pyramid_height * stencil_size.y);
    filldim3(&tile_data_size, tile_size.x - 2*border.x, tile_size.y - 2*border.y);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x), div_ceil(input_size.y, tile_data_size.y));
    unsigned int twoIterTime;
    timeInMicroSeconds(twoIterTime, (runHotspotKernel<<< grid_dims, tile_size >>>(
                                    input_size, stencil_size, device_input, device_output,
                                    pyramid_height, global_ro_data
                                    , step_div_Cap, Rx, Ry, Rz)));
    pyramid_height = 1;
    filldim3(&border, pyramid_height * stencil_size.x, pyramid_height * stencil_size.y);
    filldim3(&tile_data_size, tile_size.x - 2*border.x, tile_size.y - 2*border.y);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x), div_ceil(input_size.y, tile_data_size.y));
    unsigned int oneIterTime;
    timeInMicroSeconds(oneIterTime, (runHotspotKernel<<< grid_dims, tile_size >>>(
                                    input_size, stencil_size, device_input, device_output,
                                    pyramid_height, global_ro_data
                                    , step_div_Cap, Rx, Ry, Rz)));

#ifdef STATISTICS
/////////////////////////////////////////////////////////////////////////////////////
// Start of code to gather statistics to hone model.  Remove in final version.
////////////////////////////////////////////////////////////////////////////////////

    fprintf(stderr, "***********************************Start of a new Run****************************************\n");
    fprintf(stderr, "Data Size=%d, Tile Size=%d Iteration Count=%d\n", input_size.x, tile_size.x, iterations);

    // Precalculate the pyramid height so we can get stats on the calculated value.
    int calcMinPyramid = calcPyramidHeight(grid_dims, oneIterTime, twoIterTime);
    // Get second best for same reason.
    int secondMinPyramid = getSecond(calcMinPyramid);

    // Gather statistics to help hone model.
    double calcMinTime, secondMinTime;
    double actualMinTime = 1000000000;
    int actualMinPyramid;
    // Now let's just try them all to see what the optimal pyramid height is.
    for (int i=1; i<tile_size.x/2 - border.x; i++)
    {
        int pyramid_height = i;

        // Now we can calculate the other sizes.
        dim3 border(pyramid_height * stencil_size.x,
                    pyramid_height * stencil_size.y);
        dim3 tile_data_size(tile_size.x - 2*border.x,
                            tile_size.y - 2*border.y);
        dim3 grid_dims(div_ceil(input_size.x, tile_data_size.x),
                       div_ceil(input_size.y, tile_data_size.y));

        uint32_t time;
        timeInMicroSeconds(time, (runHotspotKernel<<< grid_dims, tile_size >>>(
                                      input_size, stencil_size, device_input, device_output,
                                      i, global_ro_data
                                      , step_div_Cap, Rx, Ry, Rz)));
        
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

    for (int i=1; i<=tile_size.x/2 - stencil_size.x; i++)
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
    filldim3(&border, pyramid_height * stencil_size.x, pyramid_height * stencil_size.y);
    filldim3(&tile_data_size, tile_size.x - 2*border.x, tile_size.y - 2*border.y);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x), div_ceil(input_size.y, tile_data_size.y));

    // Run computation
    for (int iter = 0; iter < iterations; iter += pyramid_height)
    {
        if (iter + pyramid_height > iterations)
            pyramid_height = iterations - iter;
        runHotspotKernel<<< grid_dims, tile_size >>>(
            input_size, stencil_size, device_input, device_output,
            pyramid_height, global_ro_data
            , step_div_Cap, Rx, Ry, Rz);
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
    fprintf(stderr, "Actual pyramid=%d, Actual total time=%Lu\n", i, usec2);
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
void runHotspotSetData(DTYPE *host_data, int num_elements)
{
    int num_bytes = sizeof(DTYPE) * num_elements;
    cudaMalloc((void **) &global_ro_data, num_bytes);
    cudaMemcpy(global_ro_data, host_data, num_bytes, cudaMemcpyHostToDevice);
}
