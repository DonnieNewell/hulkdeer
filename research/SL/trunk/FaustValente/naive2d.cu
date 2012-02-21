// -*- Mode: C++ ; c-file-style:"stroustrup"; indent-tabs-mode:nil; -*-

#include <stdio.h>
#include <stdlib.h>
#include "Model.cu"

#define DTYPE @DataType@

__shared__ DTYPE * inputPtr;

// Macro to get data value.
#define get(xarg,yarg)(GetInputValue(input_size, inputPtr, \
                                     ((int)((blockDim.x * blockIdx.x) + threadIdx.x)) + xarg, \
                                     ((int)((blockDim.y * blockIdx.y) + threadIdx.y)) + yarg))

#define getNew(xarg,yarg)(GetInputValue(input_size, output, \
                                     ((int)((blockDim.x * blockIdx.x) + threadIdx.x)) + xarg, \
                                     ((int)((blockDim.y * blockIdx.y) + threadIdx.y)) + yarg))

// Macro to read global read only data from within CellValue code.
#define read(offset)(ro_data[offset])

#if EdgeValue
__device__ DTYPE EdgeValue(dim3 input_size, int x, int y, DTYPE value)
{
@EdgeValue@
}
#endif

#if ConvergeValue
__device__ int converge_value;

__device__ int ConvergeValue(dim3 input_size, int x, int y, DTYPE *output @ConvergeScalarVariables@)
{
    @ConvergeValue@
}
#endif

// Take care to only read global memory is no EdgeValue is specified.
__device__ DTYPE GetInputValue(dim3 input_size, DTYPE *data, int x, int y)
{
    int ex, ey, inside;
    ex = x;
    ey = y;
    if (ex < 0) ex = 0;
    if (ey < 0) ey = 0;
    if (ex >= input_size.x) ex = input_size.x-1;
    if (ey >= input_size.y) ey = input_size.y-1;
    inside = ((x == ex) && (y == ey));
    if (inside) return data[y * input_size.x + x];
    DTYPE value = data[ey * input_size.x + ex];
#if EdgeValue
    if (!inside)
    {
        value = EdgeValue(input_size, x, y, value @ScalarVariableNames@);
    }
#endif
    return value;
}

__device__ DTYPE CellValue(dim3 input_size, int iteration, int x, int y, DTYPE *ro_data
                           @ScalarVariables@)
{
@CellValue@
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
void @FunctionName@Kernel(dim3 input_size, int iter, 
                          DTYPE *input, DTYPE *output, 
                          DTYPE *ro_data
                          @ScalarVariables@
                          @ConvergeScalarVariables@);
    }

__global__
void @FunctionName@Kernel(dim3 input_size, int iter, 
                          DTYPE *input, DTYPE *output, 
                          DTYPE *ro_data
                          @ScalarVariables@
                          @ConvergeScalarVariables@)
{
    if (threadIdx.x == 0 && threadIdx.y == 0) inputPtr = input;

    __syncthreads();    

    int bx, by, x, y;
    DTYPE value;

#if ConvergeValue
    do {
    int converge_value_result;
#endif
    // (bx, by) is the location in the input of the top left of this block.
    bx = blockIdx.x * blockDim.x;
    by = blockIdx.y * blockDim.y;
    // x is the location in the input of this thread.
    x = bx + threadIdx.x;
    y = by + threadIdx.y;
    // Make sure we are not output the data size.
    if (x >= input_size.x || y >= input_size.y) return;

    value = CellValue(input_size, iter, x, y, ro_data
                          @ScalarVariableNames@);

    output[y * input_size.x + x] = value;

#if ConvergeValue
    converge_value = @ConvergeType@;
    __syncthreads();
    converge_value_result = ConvergeValue(input_size, x, y, output @ConvergeScalarVariableNames@);
    if (@ConvergeType@) {
        if (!converge_value_result) {
            converge_value = converge_value_result;
        }
    } else {
        if (converge_value_result) {
            converge_value = converge_value_result;
        }
    }
    __syncthreads();
    } while (!converge_value);
#endif
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
                    @ScalarVariables@
                    @ConvergeScalarVariables@)
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

    dim3 grid_dims;
    filldim3(&grid_dims, div_ceil(input_size.x, tile_size.x), div_ceil(input_size.y, tile_size.y));
    fprintf(stderr, "Grid dimensions are: x=%d, y=%d\n", grid_dims.x, grid_dims.y);
    fprintf(stderr, "Tile dimensions are: x=%d, y=%d\n", tile_size.x, tile_size.y);    


/////////////////////////////////////////////////////////////////////////////////////
// Start of code to gather statistics to hone model.  Remove in final version.
////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
// End of code to gather statistics to hone model.  Remove in final version.
////////////////////////////////////////////////////////////////////////////////////

#ifdef STATISTICS
    fprintf(stderr, "***********************************Start of a new Run****************************************\n");
    fprintf(stderr, "Data Size=%d, Tile Size=%d Iteration Count=%d\n", input_size.x, tile_size.x, iterations);
    struct timeval starttime, endtime;
    unsigned int usec2;
    gettimeofday(&starttime, NULL);                                       
#endif

    // Run computation
    for (int iter = 1; iter <= iterations; iter += 1)
    {
        @FunctionName@Kernel<<< grid_dims, tile_size >>>(
            input_size, iter, device_input, device_output,
            global_ro_data
            @ScalarVariableNames@
            @ConvergeScalarVariableNames@);
#ifdef STATISTICS
        /// TEMPORARY HACK to get some debug info to make sure the kernel call succeeds.
        /// Should we leave it in, or take it out?
        cudaError_t foo = cudaGetLastError();
        if (foo != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(foo));
#endif
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
    fprintf(stderr, "Actual total time=%u\n", usec2);
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
void @FunctionName@SetData(DTYPE *host_data, int num_elements)
{
    // TEMPORARY.
    // If we want to set the cuda device number, it must be here before we call any other cuda functions.
    // cudaSetDevice(1);

    int num_bytes = sizeof(DTYPE) * num_elements;
    cudaMalloc((void **) &global_ro_data, num_bytes);
    cudaMemcpy(global_ro_data, host_data, num_bytes, cudaMemcpyHostToDevice);
}
