// -*- Mode: C++ ; c-file-style:"stroustrup"; indent-tabs-mode:nil; -*-

#include <stdio.h>
#include <stdlib.h>
#include "hotspot.h"
#include "../Model.cu"
#ifndef WIN32
	#include <sys/time.h>
#else
	#include<time.h>
#endif

#define PYRAMID_HEIGHT 1

// The size of the tile is calculated at compile time by the SL processor.
// But the data array is statically sized.
// So, make these are big as they can get.
// Chnaged to be large enough to handle fermi.
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

__device__ DTYPE CellValue(dim3 input_size, int x, int y, DTYPE *ro_data
                           , float step_div_Cap, float Rx, float Ry, float Rz) {
    float pvalue, value, term1, term2, term3, sum;
    pvalue = read(y * input_size.x + x);
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
                        DTYPE *ro_data, float step_div_Cap, float Rx,
                        float Ry, float Rz)
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
 * Store data between calls to SetData() and run().
 * This is basically a hack.
 */
static DTYPE *global_ro_data = NULL;

/**
 * this depends on all blocks being the same size
 */
static DTYPE *device_input = NULL, *device_output = NULL;

static int pyramid_height = -1; 
/**
 * Function exported to do the entire stencil computation.
 */
void runHotspot(DTYPE *host_data, int x_max, int y_max, int iterations,
                float step_div_Cap, float Rx, float Ry, float Rz, int device) {
    // User-specific parameters
    dim3 input_size(x_max, y_max);
    dim3 stencil_size(1,1);
//use the appropriate device
  int     curr_device = -1;
  
  cudaGetDevice(&curr_device);
  if (curr_device != device) {
    //changing devices, so we need to deallocate previous input/output buffers
    runHotspotCleanup();
    cudaError_t err = cudaSetDevice(device);
    if (cudaSuccess != err) {
      fprintf(stderr, "runHotspot(): couldn't select GPU index:%d.\nERROR: %s\n",
              device, cudaGetErrorString(err));
      return;
    }
  }
  
    // Host to device
    int num_bytes = input_size.x * input_size.y * sizeof(DTYPE);
  if (NULL == device_input && NULL == device_output) {
    fprintf(stderr, "allocating gpu memory.\n");
    cudaMalloc((void**) &device_output, num_bytes);
    cudaMalloc((void**) &device_input,  num_bytes);
  }
  const int newValue = 0;
  cudaMemset(static_cast<void*>(device_output), newValue, num_bytes);
    cudaMemcpy(device_input, host_data, num_bytes, cudaMemcpyHostToDevice);

    // Setup the structure that holds parameters for the application.
    // And from that, get the block size.
    char * KernelName = "runHotspotKernel";
    dim3 tile_size = initSAProps(2, input_size, stencil_size, iterations, sizeof(DTYPE), KernelName);

    dim3 border, tile_data_size, grid_dims;

    // Now we can calculate the pyramid height.
  if (-1 == pyramid_height) {  
    pyramid_height = PYRAMID_HEIGHT;
  }
    // And use the result to calculate various sizes.
    filldim3(&border,
            pyramid_height * stencil_size.x,
            pyramid_height * stencil_size.y);
    filldim3(&tile_data_size,
            tile_size.x - 2*border.x,
            tile_size.y - 2*border.y);
    filldim3(&grid_dims,
            div_ceil(input_size.x, tile_data_size.x),
            div_ceil(input_size.y, tile_data_size.y));

    // Run computation
    int tmp_pyramid_height = pyramid_height;
    for (int iter = 0; iter < iterations; iter += pyramid_height) {
        if (iter + pyramid_height > iterations)
            tmp_pyramid_height = iterations - iter;
        
        runHotspotKernel<<< grid_dims, tile_size >>>(
            input_size, stencil_size, device_input, device_output,
                tmp_pyramid_height, global_ro_data, step_div_Cap, Rx, Ry, Rz);
        DTYPE *temp = device_input;
        device_input = device_output;
        device_output = temp;
    }

    // Device to host
    cudaMemcpy(host_data, device_input, num_bytes, cudaMemcpyDeviceToHost);
    
    if (global_ro_data != NULL) {
        cudaFree(global_ro_data);
        global_ro_data = NULL;
    }
    
    //disposeSAProps(SAPs);
    SAPs = NULL;
}

void runHotspotCleanup() {
  if (device_input != NULL && device_output != NULL) {
    cudaFree(device_input);
    device_input = NULL;
    cudaFree(device_output);
    device_output = NULL;
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
