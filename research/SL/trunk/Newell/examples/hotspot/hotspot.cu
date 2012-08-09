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

// The size of the tile is calculated at compile time by the SL processor.
// But the data array is statically sized.
// So, make these are big as they can get.
// Changed to be large enough to handle fermi.
// (int)sqrt(1024) = 32
#define TILE_WIDTH  32
#define TILE_HEIGHT 32

/**
 * Block of memory shared by threads working on a single tile.
 * Contains all necessary cell values and edge values from the
 * previous iteration.
 */
__shared__ DTYPE shmem[TILE_HEIGHT][TILE_WIDTH];

__device__ DTYPE get(int x, int y) {
  return shmem[threadIdx.y + y][threadIdx.x + x];
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
  return (value + step_div_Cap * sum);
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
        float Ry, float Rz) {
  dim3 border;
  int bx, by, tx, ty, x, y, ex, ey, uidx, iter, inside;
  DTYPE value;

  // (bx, by) is the location in the input of the top left of this block.
  border.x = pyramid_height * stencil_size.x;
  border.y = pyramid_height * stencil_size.y;
  bx = blockIdx.x * (blockDim.x - 2 * border.x) - border.x;
  by = blockIdx.y * (blockDim.y - 2 * border.y) - border.y;
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
  if (ex >= input_size.x) ex = input_size.x - 1;
  if (ey >= input_size.y) ey = input_size.y - 1;

  // Get current cell value or edge value.
  uidx = ey * input_size.x + ex;
  value = input[uidx];

  inside = ((x == ex) && (y == ey));

  // Store value in shared memory for stencil calculations, and go.
  shmem[ty][tx] = value;
  iter = 0;
  border.x = border.y = 0;
  while (true) {
    __syncthreads();
    iter++;
    if (inside) {
      border.x += stencil_size.x;
      border.y += stencil_size.y;
      inside = ((tx >= border.x) && (tx < blockDim.x - border.x) &&
              (ty >= border.y) && (ty < blockDim.y - border.y));
    }
    if (inside) {
      value = CellValue(input_size, x, y, ro_data, step_div_Cap, Rx, Ry, Rz);
    }
    if (iter >= pyramid_height) {
      if (inside) {
        output[uidx] = value;
      }
      break;
    }
    __syncthreads();
    shmem[ty][tx] = value;
  }
}

void allocateCUDADataBuffers(DTYPE** &buffer1, DTYPE** &buffer2, int num_bytes) {
  //fprintf(stderr, "allocating gpu memory.\n");
  int num_devices = 0;
  cudaError_t cuda_return;
  if (cudaSuccess != cudaGetDeviceCount(&num_devices)) {
    printf("ERROR: allocateCUDADataBuffers(): device count.\n");
    return;
  }
  int current_device = -1;
  cudaGetDevice(&current_device);

  buffer1 = new DTYPE*[num_devices];
  buffer2 = new DTYPE*[num_devices];

  for (int i = 0; i < num_devices; ++i) {
    if (cudaSuccess != cudaSetDevice(i)) {
      fprintf(stderr, "allocateCUDADataBuffers(): couldn't select GPU index:%d.\nERROR: %s\n",
              i, cudaGetErrorString(cuda_return));
      return;
    }
    cuda_return = cudaMalloc(reinterpret_cast<void**> (&buffer1[i]),
            num_bytes);
    if (cudaSuccess != cuda_return) {
      printf("runHotspot: ERROR allocating output/input buffers.\n");
      return;
    }
    cuda_return = cudaMalloc(reinterpret_cast<void**> (&buffer2[i]),
            num_bytes);
    if (cudaSuccess != cuda_return) {
      printf("runHotspot: ERROR allocating output/input buffers.\n");
      return;
    }
  }

  // restore original device
  if (cudaSuccess != cudaSetDevice(current_device))
    printf("runHotspot: ERROR allocating output/input buffers.\n");
}

/**
 * Store data between calls to SetData() and run().
 * This is basically a hack.
 */
static DTYPE **cuda_global_ro_data = NULL;

/**
 * this depends on all blocks being the same size
 */
static DTYPE **device_input = NULL, **device_output = NULL;

static int pyramid_height = -1;

/**
 * Function exported to do the entire stencil computation.
 */
void runHotspot(DTYPE *host_data, int x_max, int y_max, int iterations,
        const int kPyramidHeight, float step_div_Cap, float Rx, float Ry,
        float Rz, int device) {
  // printf("runHotspot(host_data:%p, x_max:%d, y_max:%d, iterations:%d, step_div_Cap:%f, Rx:%f, Ry:%f, Rz:%f, device:%d);\n",
  //        host_data, x_max, y_max, iterations, step_div_Cap, Rx, Ry, Rz, device);
  // User-specific parameters
  const int kZero = 0;
  dim3 input_size(x_max, y_max);
  dim3 stencil_size(1, 1);
  cudaError_t cuda_error;
  //use the appropriate device
  int curr_device = -1;

  // Host to device
  int num_bytes = input_size.x * input_size.y * sizeof (DTYPE);
  if (NULL == device_input && NULL == device_output) {
    allocateCUDADataBuffers(device_input, device_output, num_bytes);
  }
  //printf("device_input:%p, device_output:%p\n", device_input, device_output);
  cudaGetDevice(&curr_device);
  if (curr_device != device) {
    cuda_error = cudaSetDevice(device);
    if (cudaSuccess != cuda_error) {
      fprintf(stderr, "runHotspot(): couldn't select GPU index:%d.\nERROR: %s\n",
              device, cudaGetErrorString(cuda_error));
      return;
    }
  }
  // printf("about to cudaMemset();\n");
  cuda_error = cudaMemset((void*) device_output[device], kZero, num_bytes);
  // printf("cudaMemset(device_output[%d]:%p, newValue:%d, num_bytes:%d)\n", device, device_output[device], newValue, num_bytes);

  cudaMemcpy(device_input[device], host_data, num_bytes, cudaMemcpyHostToDevice);
  //cuda_error = cudaMemcpy(device_input, host_data, num_bytes, cudaMemcpyHostToDevice);
  if (cudaSuccess != cuda_error) {
    printf("runHotspot: ERROR memSet %s.\n", cudaGetErrorString(cuda_error));
    exit(1);
  }

  // Setup the structure that holds parameters for the application.
  // And from that, get the block size.
  char * KernelName = "runHotspotKernel";
  // printf("initSAProps();\n");
  dim3 tile_size = initSAProps(2, input_size, stencil_size, iterations,
          sizeof (DTYPE), KernelName);
  // printf("finished initSAProps();\n");

  dim3 border, tile_data_size, grid_dims;

  // Now we can calculate the pyramid height.
  if (-1 == pyramid_height) pyramid_height = kPyramidHeight;

  // And use the result to calculate various sizes.
  filldim3(&border,
          pyramid_height * stencil_size.x,
          pyramid_height * stencil_size.y);
  filldim3(&tile_data_size,
          tile_size.x - 2 * border.x,
          tile_size.y - 2 * border.y);
  filldim3(&grid_dims,
          div_ceil(input_size.x, tile_data_size.x),
          div_ceil(input_size.y, tile_data_size.y));

  // Run computation
  int tmp_pyramid_height = pyramid_height;
  // fprintf(stdout, "runHotspotKernel(device_input:%p, device_output:%p, cuda_global_ro_data:%p);\n",
  //       device_input, device_output, cuda_global_ro_data);
  for (int iter = 0; iter < iterations; iter += pyramid_height) {
    if (iter + pyramid_height > iterations)
      tmp_pyramid_height = iterations - iter;

    runHotspotKernel << < grid_dims, tile_size >> >(
            input_size, stencil_size, device_input[device], device_output[device],
            tmp_pyramid_height, cuda_global_ro_data[device], step_div_Cap, Rx, Ry, Rz);
    DTYPE *temp = device_input[device];
    device_input[device] = device_output[device];
    device_output[device] = temp;

  }

  // Device to host
  //printf("copy results back to host_data\n");
  cudaMemcpy(host_data, device_input[device], num_bytes, cudaMemcpyDeviceToHost);

  //disposeSAProps(SAPs);
  SAPs = NULL;
}

void runHotspotCleanup() {
  int num_devices = 0;
  if (cudaSuccess != cudaGetDeviceCount(&num_devices)) {
    printf("ERROR: runHotspotCleanup: device count.\n");
    return;
  }
  if (device_input != NULL && device_output != NULL) {
    for (int i = 0; i < num_devices; ++i) {
      cudaFree(device_input[i]);
      device_input[i] = NULL;
      cudaFree(device_output[i]);
      device_output[i] = NULL;
    }
    delete [] device_input;
    device_input = NULL;
    delete [] device_output;
    device_output = NULL;
  }
  if (cuda_global_ro_data != NULL) {
    for (int i = 0; i < num_devices; ++i) {
      cudaFree(cuda_global_ro_data[i]);
      cuda_global_ro_data[i] = NULL;
    }
    delete [] cuda_global_ro_data;
    cuda_global_ro_data = NULL;
  }
}

/**
 * Store unnamed data on device.
 */
void runHotspotSetData(DTYPE *host_data, int num_elements) {
  // printf("runHotspotSetData()\n");
  int num_bytes = sizeof (DTYPE) * num_elements;
  int num_devices = 0, current_device = -1;
  cudaError_t cuda_error;
  // printf("get the number of devices...\n");
  if (cudaSuccess != cudaGetDeviceCount(&num_devices)) {
    printf("ERROR: runHotspotSetData: device count.\n");
    return;
  }
  // printf("there are %d devices on this node.\n", num_devices);
  // save current device
  cudaGetDevice(&current_device);
  cuda_global_ro_data = new DTYPE*[num_devices];
  for (int i = 0; i < num_devices; ++i) {
    // printf("setting device to %d...\n", i);
    cuda_error = cudaSetDevice(i);
    if (cudaSuccess != cuda_error) {
      fprintf(stderr, "allocateCUDADataBuffers(): couldn't select GPU index:%d.\nERROR: %s\n",
              i, cudaGetErrorString(cuda_error));
      return;
    }
    cudaMalloc((void **) &cuda_global_ro_data[i], num_bytes);
    cudaMemcpy(cuda_global_ro_data[i], host_data, num_bytes,
            cudaMemcpyHostToDevice);
  }
  // printf("setting current device back to original.\n");
  // restore original device
  cudaSetDevice(current_device);
}