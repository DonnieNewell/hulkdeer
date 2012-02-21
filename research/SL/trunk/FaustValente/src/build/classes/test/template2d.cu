// -*- Mode: C++ ; c-file-style:"stroustrup"; indent-tabs-mode:nil; -*-

#include <stdio.h>
#include <stdlib.h>
#include "Model.cu"

#define DTYPE @DataType@

// The size of the tile is calculated at compile time by the SL processor.
// But the data array is statically sized.
// So, make these as big as they can get.
// Changed to be large enough to handle Fermi.
// (int)sqrt(1024) = 32
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

// Macro to read global read only data from within CellValue code.
#define read(offset)(ro_data[offset])

__device__ DTYPE CellValue(dim3 input_size,
                           int x,
                           int y,
                           DTYPE *ro_data
                           @ScalarVariables@)
{
    @CellValue@
}

#if EdgeValue
__device__ DTYPE EdgeValue(dim3 input_size, int x, int y, DTYPE value)
{
    @EdgeValue@
}
#endif

/**
 * Each thread runs this kernel to calculate the value at one particular
 * cell in one particular iteration.
 */

extern "C" __global__
void @FunctionName@Kernel(dim3 input_size,
                          dim3 stencil_size,
                          DTYPE *input,
                          DTYPE *output,
                          int pyramid_height,
                          DTYPE *ro_data
                          @ScalarVariables@)
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
#if EdgeValue
    if (!inside)
    {
        value = EdgeValue(input_size, x, y, value @ScalarVariableNames@);
    }
#endif

#if ConvergeValue
    {
        @ConvergeValue@;
    }
#endif

    // Store value in shared memory for stencil calculations, and go.
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
            value = CellValue(input_size, x, y, ro_data
                              @ScalarVariableNames@);
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
 * Store data between calls to SetData() and run().
 * This is basically a hack.
 */
static DTYPE *global_ro_data = NULL;

/**
 * Function exported to do the entire stencil computation.
 */
void @FunctionName@(DTYPE *host_data, int x_max, int y_max, int iterations
                    @ScalarVariables@)
{
    // User-specific parameters
    dim3 input_size(x_max, y_max);
    dim3 stencil_size@StencilSize@;

    // Host to device
    DTYPE *device_input, *device_output;
    int num_bytes = input_size.x * input_size.y * sizeof(DTYPE);
    cudaMalloc((void **) &device_input, num_bytes);
    cudaMalloc((void **) &device_output, num_bytes);
    cudaMemcpy(device_input, host_data, num_bytes, cudaMemcpyHostToDevice);

    // Setup the structure that holds parameters for the application.
    // And from that, get the block size.
    char * KernelName = "@FunctionName@Kernel";
    dim3 tile_size = initSAProps(@NumDimensions@, input_size, stencil_size, iterations, sizeof(DTYPE), KernelName);

    dim3 border, tile_data_size, grid_dims;

    // Now ready for the training period.
    // Need to get some timings of small kernel runs.
    // TODO It would be faster if these could be 0 and 1 heights instead of 1 and 2.
    int pyramid_height = 2;
    filldim3(&border, pyramid_height * stencil_size.x, pyramid_height * stencil_size.y);
    filldim3(&tile_data_size, tile_size.x - 2*border.x, tile_size.y - 2*border.y);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x), div_ceil(input_size.y, tile_data_size.y));
    unsigned int twoIterTime;
    timeInMicroSeconds(twoIterTime, (@FunctionName@Kernel<<< grid_dims, tile_size >>>(
                                    input_size, stencil_size, device_input, device_output,
                                    pyramid_height, global_ro_data
                                    @ScalarVariableNames@)));
    pyramid_height = 1;
    filldim3(&border, pyramid_height * stencil_size.x, pyramid_height * stencil_size.y);
    filldim3(&tile_data_size, tile_size.x - 2*border.x, tile_size.y - 2*border.y);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x), div_ceil(input_size.y, tile_data_size.y));
    unsigned int oneIterTime;
    timeInMicroSeconds(oneIterTime, (@FunctionName@Kernel<<< grid_dims, tile_size >>>(
                                    input_size, stencil_size, device_input, device_output,
                                    pyramid_height, global_ro_data
                                    @ScalarVariableNames@)));



    // Now we can calculate the pyramid height.
    pyramid_height = calcPyramidHeight(grid_dims, oneIterTime, twoIterTime);

    // And use the result to calculate various sizes.
    filldim3(&border, pyramid_height * stencil_size.x, pyramid_height * stencil_size.y);
    filldim3(&tile_data_size, tile_size.x - 2*border.x, tile_size.y - 2*border.y);
    filldim3(&grid_dims, div_ceil(input_size.x, tile_data_size.x), div_ceil(input_size.y, tile_data_size.y));

    // Run computation
    for (int iter = 0; iter < iterations; iter += pyramid_height)
    {
        if (iter + pyramid_height > iterations)
            pyramid_height = iterations - iter;
        @FunctionName@Kernel<<< grid_dims, tile_size >>>(
            input_size, stencil_size, device_input, device_output,
            pyramid_height, global_ro_data
            @ScalarVariableNames@);
        DTYPE *temp = device_input;
        device_input = device_output;
        device_output = temp;
    }


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
void @FunctionName@SetData(DTYPE *host_data, int num_elements)
{
    int num_bytes = sizeof(DTYPE) * num_elements;
    cudaMalloc((void **) &global_ro_data, num_bytes);
    cudaMemcpy(global_ro_data, host_data, num_bytes, cudaMemcpyHostToDevice);
}
