// -*- Mode: C++ ; c-file-style:"stroustrup"; indent-tabs-mode:nil; -*-

#include "cell.h"
#include <stdio.h>
#include <stdlib.h>
#include "../Model.cu"
#ifndef WIN32
#include <sys/time.h>
#else
#include<time.h>
#endif

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


/**
 * Store data between calls to SetData() and run().
 * This is basically a hack.
 */
static DTYPE *global_ro_data = NULL;

/**
 * this depends on all blocks being the same size
 */
static DTYPE *device_input = NULL, *device_output = NULL;


__device__ DTYPE get(int x, int y, int z) {
  return shmem[threadIdx.z + z][threadIdx.y + y][threadIdx.x + x];
}

// Macro to read global read only data from within CellValue code.
#define read(offset)(ro_data[offset])

__device__ DTYPE CellValue(dim3 input_size, int x, int y, int z, DTYPE *ro_data
        , int bornMin, int bornMax, int dieMin, int dieMax) {
  int orig = get(0, 0, 0);
  int sum = 0;
  int i, j, k;
  for (i = -1; i <= 1; i++)
    for (j = -1; j <= 1; j++)
      for (k = -1; k <= 1; k++)
        sum += get(i, j, k);
  sum -= orig;
  int retval;
  if (orig > 0 && (sum <= dieMax || sum >= dieMin))
    retval = 0;
  else if (orig == 0 && (sum >= bornMin && sum <= bornMax))
    retval = 1;
  else
    retval = orig;
  return retval;
}

/**
 * Each thread runs this kernel to calculate the value at one particular
 * cell in one particular iteration.
 */

// We need to declare it C style naming.
// This avoids name mangling and allows us to get attributes about the
//  kernel call from Cuda.
// Its possible to do this with a C++ interface, but that will only
//  run on certain devices.
// This technique is older and therefore more reliable across Cuda devices.
extern "C" {
  void runCellKernel(dim3 input_size, dim3 stencil_size,
          DTYPE *input, DTYPE *output, int kPyramidHeight,
          DTYPE *ro_data
          , int bornMin, int bornMax, int dieMin, int dieMax);
}

__global__
void runCellKernel(dim3 input_size, dim3 stencil_size, DTYPE *input,
        DTYPE *output, int kPyramidHeight, DTYPE *ro_data, int bornMin,
        int bornMax, int dieMin, int dieMax) {
  dim3 border;
  int bx, by, bz, tx, ty, tz, x, y, z, ex, ey, ez, uidx, iter, inside;
  DTYPE value;

  // (bx, by, bz) is the location in the input of the top left of this block.
  border.x = kPyramidHeight * stencil_size.x;
  border.y = kPyramidHeight * stencil_size.y;
  border.z = kPyramidHeight * stencil_size.z;
  bx = blockIdx.x * (blockDim.x - 2 * border.x) - border.x;
  // These changed by Greg Faust to fix the fact that
  //     grids in CUDA cannot have 3 dimensions.
  // This parallels the same fix Jiayuan Meng used in his code for this issue.
  // UPDATE:(Donnie) There was an error in original version using the
  //  blockdim.x to get the y and z block ID's, this was changed to use
  //  gridDim.x, because blockDim gives the number of threads in a given
  //  direction, while gridDim gives the number of blocks.
  // by = blockIdx.y * (blockDim.y - 2*border.y) - border.y;
  // bz = blockIdx.z * (blockDim.z - 2*border.z) - border.z;
  int BS = blockDim.x;
  by = (blockIdx.y / gridDim.x) * (BS - 2 * border.y) - border.y;
  bz = (blockIdx.y % gridDim.x) * (BS - 2 * border.z) - border.z;

  // (x, y, z) is the location in the input of this thread.
  tx = threadIdx.x;
  ty = threadIdx.y;
  tz = threadIdx.z;
  x = bx + tx;
  y = by + ty;
  z = bz + tz;

  // (ex, ey, ez) = (x, y, z) pushed into the boundaries of the input.
  // UPDATE:(Donnie) Changed this block to ensure that we only calculate
  //    stencil values for the cells that will have all valid cells in the
  //    stencil calculation.
  ex = x;
  ey = y;
  ez = z;
  const int kValidIndex = 0;
  if (ex < kValidIndex) ex = kValidIndex;
  if (ey < kValidIndex) ey = kValidIndex;
  if (ez < kValidIndex) ez = kValidIndex;
  if (ex >= input_size.x) ex = input_size.x - 1;
  if (ey >= input_size.y) ey = input_size.y - 1;
  if (ez >= input_size.z) ez = input_size.z - 1;
  inside = ((x == ex) && (y == ey) && (z == ez));
  // Get current cell value or edge value.
  //uidx = ez + input_size.y * (ey * input_size.x + ex);
  uidx = ex + input_size.x * (ey + ez * input_size.y);
  value = input[uidx];

  // Store value in shared memory for stencil calculations, and go.
  shmem[tz][ty][tx] = value;
  iter = 0;
  border.x = border.y = border.z = 0;
  while (true) {
    __syncthreads();
    iter++;
    if (inside) {
      border.x += stencil_size.x;
      border.y += stencil_size.y;
      border.z += stencil_size.z;
      inside = ((tx >= border.x) && (tx < blockDim.x - border.x) &&
              (ty >= border.y) && (ty < blockDim.y - border.y) &&
              (tz >= border.z) && (tz < blockDim.z - border.z));
    }
    if (inside) {
      value = CellValue(input_size, x, y, z, ro_data, bornMin, bornMax,
              dieMin, dieMax);
    }
    if (iter >= kPyramidHeight) {
      if (inside) {
        output[uidx] = value;
      }
      break;
    }
    __syncthreads();
    shmem[tz][ty][tx] = value;
  }
}


/**
 * Each thread runs this kernel to calculate the value at one particular
 * cell in one particular iteration.
 */

// We need to declare it C style naming.
// This avoids name mangling and allows us to get attributes about the
//  kernel call from Cuda.
// Its possible to do this with a C++ interface, but that will only
//  run on certain devices.
// This technique is older and therefore more reliable across Cuda devices.
extern "C" {
  void runCellKernelOuter(dim3 input_size, dim3 stencil_size,
          DTYPE *input, DTYPE *output, int kPyramidHeight,
          DTYPE *ro_data
          , int bornMin, int bornMax, int dieMin, int dieMax);
}

__global__
void runCellKernelOuter(dim3 input_size, dim3 stencil_size, DTYPE *input,
        DTYPE *output, int kPyramidHeight, DTYPE *ro_data, int bornMin,
        int bornMax, int dieMin, int dieMax) {
  dim3 border;
  int bx, by, bz, tx, ty, tz, x, y, z, ex, ey, ez, uidx, iter, inside,
          in_ghost_zone(-1);
  DTYPE value;

  // (bx, by, bz) is the location in the input of the top left of this block.
  border.x = kPyramidHeight * stencil_size.x;
  border.y = kPyramidHeight * stencil_size.y;
  border.z = kPyramidHeight * stencil_size.z;
  bx = blockIdx.x * (blockDim.x - 2 * border.x) - border.x;
  // These changed by Greg Faust to fix the fact that
  //     grids in CUDA cannot have 3 dimensions.
  // This parallels the same fix Jiayuan Meng used in his code for this issue.
  // UPDATE:(Donnie) There was an error in original version using the
  //  blockdim.x to get the y and z block ID's, this was changed to use
  //  gridDim.x, because blockDim gives the number of threads in a given
  //  direction, while gridDim gives the number of blocks.
  // by = blockIdx.y * (blockDim.y - 2*border.y) - border.y;
  // bz = blockIdx.z * (blockDim.z - 2*border.z) - border.z;
  int BS = blockDim.x;
  by = (blockIdx.y / gridDim.x) * (BS - 2 * border.y) - border.y;
  bz = (blockIdx.y % gridDim.x) * (BS - 2 * border.z) - border.z;

  // (x, y, z) is the location in the input of this thread.
  tx = threadIdx.x;
  ty = threadIdx.y;
  tz = threadIdx.z;
  x = bx + tx;
  y = by + ty;
  z = bz + tz;

  // (ex, ey, ez) = (x, y, z) pushed into the boundaries of the input.
  // UPDATE:(Donnie) Changed this block to ensure that we only calculate
  //    stencil values for the cells that will have all valid cells in the
  //    stencil calculation.
  ex = x;
  ey = y;
  ez = z;
  const int kValidIndex = 0;
  if (ex < kValidIndex) ex = kValidIndex;
  if (ey < kValidIndex) ey = kValidIndex;
  if (ez < kValidIndex) ez = kValidIndex;
  if (ex >= input_size.x) ex = input_size.x - 1;
  if (ey >= input_size.y) ey = input_size.y - 1;
  if (ez >= input_size.z) ez = input_size.z - 1;
  
  inside = ((x == ex) && (y == ey) && (z == ez));
  
  // Get current cell value or edge value.
  //uidx = ez + input_size.y * (ey * input_size.x + ex);
  uidx = ex + input_size.x * (ey + ez * input_size.y);
  value = input[uidx];

  // Store value in shared memory for stencil calculations, and go.
  shmem[tz][ty][tx] = value;
  
  const int kGhostZ = kPyramidHeight * stencil_size.z;
  const int kGhostY = kPyramidHeight * stencil_size.y;
  const int kGhostX = kPyramidHeight * stencil_size.x;
  // front and back
  in_ghost_zone = ((ez < kGhostZ) || (ez > input_size.z - kGhostZ));
  // top and bottom
  in_ghost_zone = in_ghost_zone || ((ez >= kGhostZ) && (ez < input_size.z - kGhostZ) &&
          (ey >= input_size.y - kGhostY) && (ey < input_size.y - border.y) &&
          (ex >= border.x) && (ex < input_size.x - border.x));
  in_ghost_zone = in_ghost_zone || ((ez >= kGhostZ) && (ez < input_size.z - kGhostZ) &&
          (ey >= border.y) && (ey < kGhostY) &&
          (ex >= border.x) && (ex < input_size.x - border.x));
  // left and right
  in_ghost_zone = in_ghost_zone || ((ez >= kGhostZ) && (ez < input_size.z - kGhostZ) &&
          (ey >= kGhostY) && (ey < input_size.y - kGhostY) &&
          (ex >= border.x) && (ex < kGhostX));
  in_ghost_zone = in_ghost_zone || ((ez >= kGhostZ) && (ez < input_size.z - kGhostZ) &&
          (ey >= kGhostY) && (ey < input_size.y - kGhostY) &&
          (ex >= input_size.x - kGhostX) && (ex < input_size.x - border.x));
  
  inside = inside && in_ghost_zone;
  iter = 0;
  border.x = border.y = border.z = 0;
  while (true) {
    __syncthreads();
    iter++;
    if (inside) {
      border.x += stencil_size.x;
      border.y += stencil_size.y;
      border.z += stencil_size.z;
      inside = ((tx >= border.x) && (tx < blockDim.x - border.x) &&
              (ty >= border.y) && (ty < blockDim.y - border.y) &&
              (tz >= border.z) && (tz < blockDim.z - border.z));
    }
    if (inside) {
      value = CellValue(input_size, x, y, z, ro_data, bornMin, bornMax,
              dieMin, dieMax);
    }
    if (iter >= kPyramidHeight) {
      if (inside) {
        output[uidx] = value;
      }
      break;
    }
    __syncthreads();
    shmem[tz][ty][tx] = value;
  }
}
/**
 * Each thread runs this kernel to calculate the value at one particular
 * cell in one particular iteration.
 */

// We need to declare it C style naming.
// This avoids name mangling and allows us to get attributes about the
//  kernel call from Cuda.
// Its possible to do this with a C++ interface, but that will only
//  run on certain devices.
// This technique is older and therefore more reliable across Cuda devices.
extern "C" {
  void runCellKernelInner(dim3 input_size, dim3 stencil_size,
          DTYPE *input, DTYPE *output, int kPyramidHeight,
          DTYPE *ro_data
          , int bornMin, int bornMax, int dieMin, int dieMax);
}

__global__
void runCellKernelInner(dim3 input_size, dim3 stencil_size, DTYPE *input,
        DTYPE *output, int kPyramidHeight, DTYPE *ro_data, int bornMin,
        int bornMax, int dieMin, int dieMax) {
  dim3 border;
  int bx, by, bz, tx, ty, tz, x, y, z, ex, ey, ez, uidx, iter, inside,
          in_inner_zone(-1);
  DTYPE value;

  // (bx, by, bz) is the location in the input of the top left of this block.
  border.x = kPyramidHeight * stencil_size.x;
  border.y = kPyramidHeight * stencil_size.y;
  border.z = kPyramidHeight * stencil_size.z;
  bx = blockIdx.x * (blockDim.x - 2 * border.x) - border.x;
  // These changed by Greg Faust to fix the fact that
  //     grids in CUDA cannot have 3 dimensions.
  // This parallels the same fix Jiayuan Meng used in his code for this issue.
  // UPDATE:(Donnie) There was an error in original version using the
  //  blockdim.x to get the y and z block ID's, this was changed to use
  //  gridDim.x, because blockDim gives the number of threads in a given
  //  direction, while gridDim gives the number of blocks.
  // by = blockIdx.y * (blockDim.y - 2*border.y) - border.y;
  // bz = blockIdx.z * (blockDim.z - 2*border.z) - border.z;
  int BS = blockDim.x;
  by = (blockIdx.y / gridDim.x) * (BS - 2 * border.y) - border.y;
  bz = (blockIdx.y % gridDim.x) * (BS - 2 * border.z) - border.z;

  // (x, y, z) is the location in the input of this thread.
  tx = threadIdx.x;
  ty = threadIdx.y;
  tz = threadIdx.z;
  x = bx + tx;
  y = by + ty;
  z = bz + tz;

  // (ex, ey, ez) = (x, y, z) pushed into the boundaries of the input.
  // UPDATE:(Donnie) Changed this block to ensure that we only calculate
  //    stencil values for the cells that will have all valid cells in the
  //    stencil calculation.
  ex = x;
  ey = y;
  ez = z;
  const int kValidIndex = 0;
  if (ex < kValidIndex) ex = kValidIndex;
  if (ey < kValidIndex) ey = kValidIndex;
  if (ez < kValidIndex) ez = kValidIndex;
  if (ex >= input_size.x) ex = input_size.x - 1;
  if (ey >= input_size.y) ey = input_size.y - 1;
  if (ez >= input_size.z) ez = input_size.z - 1;
  
  inside = ((x == ex) && (y == ey) && (z == ez));
  
  // Get current cell value or edge value.
  //uidx = ez + input_size.y * (ey * input_size.x + ex);
  uidx = ex + input_size.x * (ey + ez * input_size.y);
  value = input[uidx];

  // Store value in shared memory for stencil calculations, and go.
  shmem[tz][ty][tx] = value;
  
  const int kGhostZ = kPyramidHeight * stencil_size.z;
  const int kGhostY = kPyramidHeight * stencil_size.y;
  const int kGhostX = kPyramidHeight * stencil_size.x;
  // front and back
  in_inner_zone = (ez >= kGhostZ) && (ez < input_size.z - kGhostZ) &&
          (ey >= kGhostY) && (ey < input_size.y - kGhostY) &&
          (ex >= kGhostX) && (ex < input_size.x - kGhostX);
  inside = inside && in_inner_zone;
  iter = 0;
  border.x = border.y = border.z = 0;
  while (true) {
    __syncthreads();
    iter++;
    if (inside) {
      border.x += stencil_size.x;
      border.y += stencil_size.y;
      border.z += stencil_size.z;
      inside = ((tx >= border.x) && (tx < blockDim.x - border.x) &&
              (ty >= border.y) && (ty < blockDim.y - border.y) &&
              (tz >= border.z) && (tz < blockDim.z - border.z));
    }
    if (inside) {
      value = CellValue(input_size, x, y, z, ro_data, bornMin, bornMax,
              dieMin, dieMax);
    }
    if (iter >= kPyramidHeight) {
      if (inside) {
        output[uidx] = value;
      }
      break;
    }
    __syncthreads();
    shmem[tz][ty][tx] = value;
  }
}

/**
 * Function exported to do the entire stencil computation.
 */
void runCell(DTYPE *host_data, int x_max, int y_max, int z_max, int iterations,
        const int kPyramidHeight, int bornMin, int bornMax, int dieMin,
        int dieMax, int device) {
  // User-specific parameters
  dim3 input_size(x_max, y_max, z_max);
  dim3 stencil_size(1, 1, 1);
  //use the appropriate device
  int curr_device = -1;
  struct timeval start, end;

  cudaGetDevice(&curr_device);
  if (curr_device != device) {
    //changing devices, so we need to deallocate previous input/output buffers
    runCellCleanup();
    cudaError_t err = cudaSetDevice(device);
    if (cudaSuccess != err) {
      fprintf(stderr, "runCell(): couldn't select GPU index:%d.\nERROR: %s\n",
              device, cudaGetErrorString(err));
      return;
    }
  }

  // Allocate CUDA arrays in device memory 
  int num_bytes = input_size.x * input_size.y * input_size.z * sizeof (DTYPE);
  if (NULL == device_input && NULL == device_output) {
    //fprintf(stderr, "allocating gpu memory.\n");
    cudaMalloc((void**) &device_output, num_bytes);
    cudaMalloc((void**) &device_input, num_bytes);
  }
  //cudaMemset(static_cast<void*> (device_output), newValue, num_bytes);
  cudaMemcpy(device_input, host_data, num_bytes, cudaMemcpyHostToDevice);

  // Setup the structure that holds parameters for the application.
  // And from that, get the block size.
  char* KernelName = "runCellKernel";
  dim3 tile_size = initSAProps(3, input_size, stencil_size, iterations,
          sizeof (DTYPE), KernelName);
  dim3 border,
          tile_data_size,
          grid_dims;

  // And use the result to calculate various sizes.
  filldim3(&border,
          kPyramidHeight * stencil_size.x,
          kPyramidHeight * stencil_size.y,
          kPyramidHeight * stencil_size.z);
  filldim3(&tile_data_size,
          tile_size.x - 2 * border.x,
          tile_size.y - 2 * border.y,
          tile_size.z - 2 * border.z);
  filldim3(&grid_dims,
          div_ceil(input_size.x, tile_data_size.x),
          //      div_ceil(input_size.y, tile_data_size.y),
          //      div_ceil(input_size.z, tile_data_size.z));
          div_ceil(input_size.y, tile_data_size.y) *
          div_ceil(input_size.z, tile_data_size.z)); //*/

  gettimeofday(&start, NULL);
  // Run computation
  int tmp_pyramid_height = kPyramidHeight;
  for (int iter = 0; iter < iterations; iter += kPyramidHeight) {
    if (iter + kPyramidHeight > iterations)
      tmp_pyramid_height = iterations - iter;
    runCellKernel <<<grid_dims, tile_size>>>( input_size, stencil_size,
            device_input, device_output, tmp_pyramid_height, global_ro_data,
            bornMin, bornMax, dieMin, dieMax);

    DTYPE *temp = device_input;
    device_input = device_output;
    device_output = temp;
  }
  gettimeofday(&end, NULL);

  // Device to host
  cudaMemcpy(host_data, device_input, num_bytes, cudaMemcpyDeviceToHost);

  if (global_ro_data != NULL) {
    cudaFree(global_ro_data);
    global_ro_data = NULL;
  }

  //disposeSAProps(SAPs);
  SAPs = NULL;
}

/**
 * Function exported to do the entire stencil computation.
 */
void runCellInner(DTYPE *host_data, int x_max, int y_max, int z_max, int iterations,
        const int kPyramidHeight, int bornMin, int bornMax, int dieMin,
        int dieMax, int device) {
  // User-specific parameters
  dim3 input_size(x_max, y_max, z_max);
  dim3 stencil_size(1, 1, 1);
  //use the appropriate device
  int curr_device = -1;
  struct timeval start, end;

  cudaGetDevice(&curr_device);
  if (curr_device != device) {
    //changing devices, so we need to deallocate previous input/output buffers
    runCellCleanup();
    cudaError_t err = cudaSetDevice(device);
    if (cudaSuccess != err) {
      fprintf(stderr, "runCell(): couldn't select GPU index:%d.\nERROR: %s\n",
              device, cudaGetErrorString(err));
      return;
    }
  }

  // Allocate CUDA arrays in device memory 
  int num_bytes = input_size.x * input_size.y * input_size.z * sizeof (DTYPE);
  if (NULL == device_input && NULL == device_output) {
    //fprintf(stderr, "allocating gpu memory.\n");
    cudaMalloc((void**) &device_output, num_bytes);
    cudaMalloc((void**) &device_input, num_bytes);
  }
  //cudaMemset(static_cast<void*> (device_output), newValue, num_bytes);
  cudaMemcpy(device_input, host_data, num_bytes, cudaMemcpyHostToDevice);

  // Setup the structure that holds parameters for the application.
  // And from that, get the block size.
  char* KernelName = "runCellKernelInner";
  dim3 tile_size = initSAProps(3, input_size, stencil_size, iterations,
          sizeof (DTYPE), KernelName);
  dim3 border,
          tile_data_size,
          grid_dims;

  // And use the result to calculate various sizes.
  filldim3(&border,
          kPyramidHeight * stencil_size.x,
          kPyramidHeight * stencil_size.y,
          kPyramidHeight * stencil_size.z);
  filldim3(&tile_data_size,
          tile_size.x - 2 * border.x,
          tile_size.y - 2 * border.y,
          tile_size.z - 2 * border.z);
  filldim3(&grid_dims,
          div_ceil(input_size.x, tile_data_size.x),
          //      div_ceil(input_size.y, tile_data_size.y),
          //      div_ceil(input_size.z, tile_data_size.z));
          div_ceil(input_size.y, tile_data_size.y) *
          div_ceil(input_size.z, tile_data_size.z)); //*/

  gettimeofday(&start, NULL);
  // Run computation
  int tmp_pyramid_height = kPyramidHeight;
  for (int iter = 0; iter < iterations; iter += kPyramidHeight) {
    if (iter + kPyramidHeight > iterations)
      tmp_pyramid_height = iterations - iter;
    runCellKernelInner <<<grid_dims, tile_size>>>( input_size, stencil_size,
            device_input, device_output, tmp_pyramid_height, global_ro_data,
            bornMin, bornMax, dieMin, dieMax);

    DTYPE *temp = device_input;
    device_input = device_output;
    device_output = temp;
  }
  gettimeofday(&end, NULL);

  // Device to host
  cudaMemcpy(host_data, device_input, num_bytes, cudaMemcpyDeviceToHost);

  if (global_ro_data != NULL) {
    cudaFree(global_ro_data);
    global_ro_data = NULL;
  }

  //disposeSAProps(SAPs);
  SAPs = NULL;
}

/**
 * Function exported to do the entire stencil computation.
 */
void runCellOuter(DTYPE *host_data, int x_max, int y_max, int z_max, int iterations,
        const int kPyramidHeight, int bornMin, int bornMax, int dieMin,
        int dieMax, int device) {
  // User-specific parameters
  dim3 input_size(x_max, y_max, z_max);
  dim3 stencil_size(1, 1, 1);
  //use the appropriate device
  int curr_device = -1;
  struct timeval start, end;

  cudaGetDevice(&curr_device);
  if (curr_device != device) {
    //changing devices, so we need to deallocate previous input/output buffers
    runCellCleanup();
    cudaError_t err = cudaSetDevice(device);
    if (cudaSuccess != err) {
      fprintf(stderr, "runCell(): couldn't select GPU index:%d.\nERROR: %s\n",
              device, cudaGetErrorString(err));
      return;
    }
  }

  // Allocate CUDA arrays in device memory 
  int num_bytes = input_size.x * input_size.y * input_size.z * sizeof (DTYPE);
  if (NULL == device_input && NULL == device_output) {
    //fprintf(stderr, "allocating gpu memory.\n");
    cudaMalloc((void**) &device_output, num_bytes);
    cudaMalloc((void**) &device_input, num_bytes);
  }
  //cudaMemset(static_cast<void*> (device_output), newValue, num_bytes);
  cudaMemcpy(device_input, host_data, num_bytes, cudaMemcpyHostToDevice);

  // Setup the structure that holds parameters for the application.
  // And from that, get the block size.
  char* KernelName = "runCellKernelOuter";
  dim3 tile_size = initSAProps(3, input_size, stencil_size, iterations,
          sizeof (DTYPE), KernelName);
  dim3 border,
          tile_data_size,
          grid_dims;

  // And use the result to calculate various sizes.
  filldim3(&border,
          kPyramidHeight * stencil_size.x,
          kPyramidHeight * stencil_size.y,
          kPyramidHeight * stencil_size.z);
  filldim3(&tile_data_size,
          tile_size.x - 2 * border.x,
          tile_size.y - 2 * border.y,
          tile_size.z - 2 * border.z);
  filldim3(&grid_dims,
          div_ceil(input_size.x, tile_data_size.x),
          //      div_ceil(input_size.y, tile_data_size.y),
          //      div_ceil(input_size.z, tile_data_size.z));
          div_ceil(input_size.y, tile_data_size.y) *
          div_ceil(input_size.z, tile_data_size.z)); //*/

  gettimeofday(&start, NULL);
  // Run computation
  int tmp_pyramid_height = kPyramidHeight;
  for (int iter = 0; iter < iterations; iter += kPyramidHeight) {
    if (iter + kPyramidHeight > iterations)
      tmp_pyramid_height = iterations - iter;
    runCellKernelOuter <<<grid_dims, tile_size>>>( input_size, stencil_size,
            device_input, device_output, tmp_pyramid_height, global_ro_data,
            bornMin, bornMax, dieMin, dieMax);

    DTYPE *temp = device_input;
    device_input = device_output;
    device_output = temp;
  }
  gettimeofday(&end, NULL);

  // Device to host
  cudaMemcpy(host_data, device_input, num_bytes, cudaMemcpyDeviceToHost);

  if (global_ro_data != NULL) {
    cudaFree(global_ro_data);
    global_ro_data = NULL;
  }

  //disposeSAProps(SAPs);
  SAPs = NULL;
}

void runCellCleanup() {
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
void runCellSetData(DTYPE *host_data, int num_elements) {
  int num_bytes = sizeof (DTYPE) * num_elements;
  cudaMalloc((void **) &global_ro_data, num_bytes);
  cudaMemcpy(global_ro_data, host_data, num_bytes, cudaMemcpyHostToDevice);
}
