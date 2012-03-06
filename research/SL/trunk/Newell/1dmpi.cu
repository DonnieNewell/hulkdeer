// -*- Mode: C++ ; c-file-style:"stroustrup"; indent-tabs-mode:nil; -*-

#include <stdio.h>
#include <stdlib.h>
#include "Model.cu"

#define DTYPE int

// The size of the tile is calculated at compile time by the SL processor.
// But the data array is statically sized.
// So, make these are big as they can get.
#define TILE_WIDTH 1024

/**
 * Block of memory shared by threads working on a single tile.
 * Contains all necessary cell values and edge values from the
 * previous iteration.
 */
__shared__ DTYPE shmem[TILE_WIDTH];

__device__ DTYPE get(int x)
{
    return shmem[threadIdx.x+x];
}

// Macro to read global read only data from within CellValue code.
#define read(offset)(ro_data[offset])

__device__ DTYPE CellValue(dim3 input_size, int iteration, int x, DTYPE *ro_data
                           , int bornMin, int bornMax, int dieMin, int dieMax)
{
    int orig = get(0);
    int sum = 0;
    int i, j, k;
    for (i = -1; i <= 1; i++)
        sum += get(i);
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
void runCellKernel(dim3 input_size, dim3 stencil_size, int iters, 
                          DTYPE *input, DTYPE *output, int pyramid_height,
                          DTYPE *ro_data
                          , int bornMin, int bornMax, int dieMin, int dieMax);
    }

__global__
void runCellKernel(dim3 input_size, dim3 stencil_size, int iters,
                          DTYPE *input, DTYPE *output, int pyramid_height,
                          DTYPE *ro_data
                          , int bornMin, int bornMax, int dieMin, int dieMax)
{
    dim3 border;
    int bx, tx, x, ex, uidx, iter, inside;
    DTYPE value;

    // bx is the location in the input of the left of this block.
    border.x = pyramid_height * stencil_size.x;
    bx = blockIdx.x * (blockDim.x - 2*border.x) - border.x;
    // x is the location in the input of this thread.
    tx = threadIdx.x;
    x = bx + tx;

    // ex = x pushed into the boundaries of the input.
    ex = x;
    if (ex < 0) ex = 0;
    if (ex >= input_size.x) ex = input_size.x-1;

    // Get current cell value or edge value.
    uidx = ex;
    value = input[uidx];
    inside = ((x == ex));

    // Store value in shared memory for stencil calculations, and go.
    shmem[tx] = value;
    iter = 0;
    border.x = 0;
    while (true)
    {
        __syncthreads();
        iter++;
        if (inside)
        {
            border.x += stencil_size.x;
            inside = ((tx >= border.x) && (tx < blockDim.x-border.x));
        }
        if (inside)
        {
            value = CellValue(input_size, iters + iter, x, ro_data
                              , bornMin, bornMax, dieMin, dieMax);
        }
        if (iter >= pyramid_height)
        {
            if (inside)
                output[uidx] = value;
            break;
        }
        __syncthreads();
        shmem[tx] = value;
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
void runCell(DTYPE *host_data, int x_max, int iterations, int rank, int size
                    , int bornMin, int bornMax, int dieMin, int dieMax)
{
    // User-specific parameters
    dim3 input_size(x_max);
    dim3 stencil_size(1);

    // Host to device
    DTYPE *device_input, *device_output;
    int num_bytes = input_size.x * sizeof(DTYPE);


    char *dtype;
    dtype = #DTYPE;
    MPI_Datatype mpi_datatype;
    switch (*dtype) {
    case 'f':
        mpi_datatype = MPI_FLOAT;
    case 'i';
        mpi_datatype = MPI_INT;
    case 'd';
        mpi_datatype = MPI_DOUBLE;
    default:
        exit(1);
    }
    DTYPE *local_data = new DTYPE[x_max];

    MPI_Scatter(host_data, x_max / , mpi_datatype,
                local_data, x_max, mpi_datatype,
                0, MPI_COMM_WORLD);

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
    dim3 tile_size = initSAProps(1, input_size, stencil_size, iterations, sizeof(DTYPE), KernelName);

    dim3 border, tile_data_size, grid_dims;

    // Now ready for the training period.
    // Need to get some timings of small kernel runs.
    // TODO It would be faster if these could be 0 and 1 heights instead of 1 and 2.
    int pyramid_height = 2;
    filldim3(&border, pyramid_height * stencil_size.x);
    filldim3(&tile_data_size, tile_size.x - 2*border.x);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x));
    unsigned int twoIterTime;
    timeInMicroSeconds(twoIterTime, (runCellKernel<<< grid_dims, tile_size >>>(
                                         input_size, stencil_size, 0, device_input, device_output,
                                    pyramid_height, global_ro_data
                                    , bornMin, bornMax, dieMin, dieMax)));
    pyramid_height = 1;
    filldim3(&border, pyramid_height * stencil_size.x);
    filldim3(&tile_data_size, tile_size.x - 2*border.x);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x));
    unsigned int oneIterTime;
    timeInMicroSeconds(oneIterTime, (runCellKernel<<< grid_dims, tile_size >>>(
                                    input_size, stencil_size, 0, device_input, device_output,
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
    uint32_t pyr1time;
    // Now let's just try them all to see what the optimal pyramid height is.
    for (int i=1; i<tile_size.x/(2 * stencil_size.x); i++)
    {
        int pyramid_height = i;
        if (pyramid_height > iterations) break;

        // Now we can calculate the other sizes.
        dim3 border(pyramid_height * stencil_size.x);
        dim3 tile_data_size(tile_size.x - 2*border.x);
        dim3 grid_dims(div_ceil(input_size.x, tile_data_size.x));

        uint32_t time;
        timeInMicroSeconds(time, (runCellKernel<<< grid_dims, tile_size >>>(
                                      input_size, stencil_size, 0, device_input, device_output,
                                      i, global_ro_data
                                      , bornMin, bornMax, dieMin, dieMax)));
        
        double timePer = ((double)time)/i;
        if (i == 1) pyr1time = time;
        if (i == calcMinPyramid) calcMinTime = timePer;
        if (i == secondMinPyramid) secondMinTime = timePer;
        // The multiples of 16 are magically good.  Let's see how we compare against non-magic numbers!
        // if (i % 16 == 00) continue;
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
    fprintf(stderr, "Size %d Pyramid1Time %u BestHeight %d BestTime %4.2f CalcHeight %d %%Slowdown %4.2f CalcSecond %d %%Slowdown %4.2f MinSlowdown %4.2f\n", 
            input_size.x, pyr1time, actualMinPyramid, actualMinTime, calcMinPyramid, firstError, secondMinPyramid, secondError, MIN(firstError, secondError));

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

    filldim3(&border, pyramid_height * stencil_size.x);
    filldim3(&tile_data_size, tile_size.x - 2*border.x);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x));

    // Run computation
    for (int iter = 0; iter < iterations; iter += pyramid_height)
    {
        if (iter + pyramid_height > iterations)
            pyramid_height = iterations - iter;

        runCellKernel<<< grid_dims, tile_size >>>(
            input_size, stencil_size, iter, device_input, device_output,
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
    // TEMPORARY.
    // If we want to set the cuda device number, it must be here before we call any other cuda functions.
    // cudaSetDevice(1);

    int num_bytes = sizeof(DTYPE) * num_elements;
    cudaMalloc((void **) &global_ro_data, num_bytes);
    cudaMemcpy(global_ro_data, host_data, num_bytes, cudaMemcpyHostToDevice);
}
