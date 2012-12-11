/*
  Copyright 2012 Donald Newell
 */

#include "ompCell.h"
#include "../comm.h"
#ifndef WIN32
#include <sys/time.h>
#else
#include<time.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <algorithm>

// The size of the tile is calculated at compile time by the SL processor.
// But the data array is statically sized.
// So, make these are big as they can get.
// Changed to be large enough for fermi
// (int)cube_rt(1024) = 10
#define TILE_WIDTH  10
#define TILE_HEIGHT 10
#define TILE_DEPTH  10

typedef struct dim {
  int x;
  int y;
  int z;
} dim3;

static double memcpy_time = 0.0;
void copyHostData(DTYPE* src_data, dim3 size, dim3 border, DTYPE* dest_data);
void copyOuterHostData(DTYPE* src_data, dim3 size, dim3 border, DTYPE* dest_data);
void copyInnerHostData(DTYPE* src_data, dim3 size, dim3 border, DTYPE* dest_data);
double secondsElapsed(struct timeval start, struct timeval stop);

double getMemcpyTime() {
  return memcpy_time;
}

// Macro to read global read only data from within CellValue code.
#define read(offset)(ro_data[offset])

DTYPE OMPCellValue(dim3 input_size, int x, int y, int z, DTYPE *input,
        int bornMin, int bornMax, int dieMin, int dieMax, dim3 border) {
  int uidx = x + input_size.x * (y + z * input_size.y);
  int orig = input[uidx];
  int sum = 0;
  int i, j, k;
  for (i = -1; i <= 1; i++) {
    for (j = -1; j <= 1; j++) {
      for (k = -1; k <= 1; k++) {
        uidx = x + k + input_size.x * (y + j + (z + i) * input_size.y);
        sum += input[uidx];
      }
    }
  }
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
// This avoids name mangling and allows us to get attributes about the kernel
// call from Cuda. Its possible to do this with a C++ interface, but that will
// only run on certain devices. This technique is older and therefore more
// reliable across Cuda devices.
extern "C" {
  void runOMPCellKernelOuter(dim3 input_size, dim3 stencil_size,
          DTYPE *input, DTYPE *output, int pyramid_height,
          DTYPE *ro_data
          , int bornMin, int bornMax, int dieMin, int dieMax);
}

void runOMPCellKernelOuter(dim3 input_size, dim3 stencil_size, DTYPE *input,
        DTYPE *output, int pyramid_height, DTYPE *ro_data,
        int bornMin, int bornMax, int dieMin, int dieMax) {
  dim3 border;
  border.x = border.y = border.z = 0;
  const int kGhostZ = pyramid_height * stencil_size.z;
  const int kGhostY = pyramid_height * stencil_size.y;
  const int kGhostX = pyramid_height * stencil_size.x;
  for (int iter = 0; iter < pyramid_height; ++iter) {
    border.x += stencil_size.x;
    border.y += stencil_size.y;
    border.z += stencil_size.z;
    int uidx = -1;
    // (x, y, z) is the location in the input of this thread.
    // TODO 6 loops to cover all of the gz faces
    // front face
#pragma omp parallel for private(uidx) shared(output)
    for (int z = border.z; z < kGhostZ; ++z) {
      for (int y = border.y; y < input_size.y - border.y; ++y) {
        for (int x = border.x; x < input_size.x - border.x; ++x) {
          // Get current cell value or edge value.
          uidx = x + input_size.x * (y + z * input_size.y);
          output[uidx] = OMPCellValue(input_size, x, y, z, input,
                  bornMin, bornMax, dieMin, dieMax, border);
        }
      }
    }

    // back face
#pragma omp parallel for private(uidx) shared(output)
    for (int z = input_size.z - kGhostZ;
            z < input_size.z - border.z;
            ++z) {
      for (int y = border.y; y < input_size.y - border.y; ++y) {
        for (int x = border.x; x < input_size.x - border.x; ++x) {
          // Get current cell value or edge value.
          uidx = x + input_size.x * (y + z * input_size.y);
          output[uidx] = OMPCellValue(input_size, x, y, z, input,
                  bornMin, bornMax, dieMin, dieMax, border);
        }
      }
    }

#ifndef SLAB
    // top face
#pragma omp parallel for private(uidx) shared(output)
    for (int z = kGhostZ; z < input_size.z - kGhostZ; ++z) {
      for (int y = input_size.y - kGhostY; y < input_size.y - border.y; ++y) {
        for (int x = border.x; x < input_size.x - border.x; ++x) {
          // Get current cell value or edge value.
          uidx = x + input_size.x * (y + z * input_size.y);
          output[uidx] = OMPCellValue(input_size, x, y, z, input,
                  bornMin, bornMax, dieMin, dieMax, border);
        }
      }
    }

    // bottom face
#pragma omp parallel for private(uidx) shared(output)
    for (int z = kGhostZ; z < input_size.z - kGhostZ; ++z) {
      for (int y = border.y; y < kGhostY; ++y) {
        for (int x = border.x; x < input_size.x - border.x; ++x) {
          // Get current cell value or edge value.
          uidx = x + input_size.x * (y + z * input_size.y);
          output[uidx] = OMPCellValue(input_size, x, y, z, input,
                  bornMin, bornMax, dieMin, dieMax, border);
        }
      }
    }

    // left face
#pragma omp parallel for private(uidx) shared(output)
    for (int z = kGhostZ; z < input_size.z - kGhostZ; ++z) {
      for (int y = kGhostY; y < input_size.y - kGhostY; ++y) {
        for (int x = border.x; x < kGhostX; ++x) {
          // Get current cell value or edge value.
          uidx = x + input_size.x * (y + z * input_size.y);
          output[uidx] = OMPCellValue(input_size, x, y, z, input,
                  bornMin, bornMax, dieMin, dieMax, border);
        }
      }
    }

    // right face
#pragma omp parallel for private(uidx) shared(output)
    for (int z = kGhostZ; z < input_size.z - kGhostZ; ++z) {
      for (int y = kGhostY; y < input_size.y - kGhostY; ++y) {
        for (int x = input_size.x - kGhostX; x < input_size.x - border.x; ++x) {
          // Get current cell value or edge value.
          uidx = x + input_size.x * (y + z * input_size.y);
          output[uidx] = OMPCellValue(input_size, x, y, z, input,
                  bornMin, bornMax, dieMin, dieMax, border);
        }
      }
    }
#endif
  }
}

// We need to declare it C style naming.
// This avoids name mangling and allows us to get attributes about the kernel
// call from Cuda. Its possible to do this with a C++ interface, but that will
// only run on certain devices. This technique is older and therefore more
// reliable across Cuda devices.
extern "C" {
  void runOMPCellKernelInner(dim3 input_size, dim3 stencil_size,
          DTYPE *input, DTYPE *output, int pyramid_height,
          DTYPE *ro_data
          , int bornMin, int bornMax, int dieMin, int dieMax);
}

void runOMPCellKernelInner(dim3 input_size, dim3 stencil_size, DTYPE *input,
        DTYPE *output, int pyramid_height, DTYPE *ro_data,
        int bornMin, int bornMax, int dieMin, int dieMax) {
  dim3 border;
  border.x = border.y = border.z = 0;
  const int kGhostZ = pyramid_height * stencil_size.z;
  const int kGhostY = pyramid_height * stencil_size.y;
  const int kGhostX = pyramid_height * stencil_size.x;
  for (int iter = 0; iter < pyramid_height; ++iter) {
    border.x += stencil_size.x;
    border.y += stencil_size.y;
    border.z += stencil_size.z;
    int uidx = -1;
    // (x, y, z) is the location in the input of this thread.
    // TODO 6 loops to cover all of the gz faces
    // front face
#pragma omp parallel for private(uidx) shared(output)
    for (int z = kGhostZ; z < input_size.z - kGhostZ; ++z) {
      for (int y = border.y; y < input_size.y - border.y; ++y) {
        for (int x = border.x; x < input_size.x - border.x; ++x) {
          // Get current cell value or edge value.
          uidx = x + input_size.x * (y + z * input_size.y);
          output[uidx] = OMPCellValue(input_size, x, y, z, input,
                  bornMin, bornMax, dieMin, dieMax, border);
        }
      }
    }
  }
}

extern "C" {
  void runOMPCellKernel(dim3 input_size, dim3 stencil_size,
          DTYPE *input, DTYPE *output, int pyramid_height,
          DTYPE *ro_data
          , int bornMin, int bornMax, int dieMin, int dieMax);
}

void runOMPCellKernel(dim3 input_size, dim3 stencil_size, DTYPE *input,
        DTYPE *output, int pyramid_height, DTYPE *ro_data,
        int bornMin, int bornMax, int dieMin, int dieMax) {
  dim3 border;
  border.x = border.y = border.z = 0;
  for (int iter = 0; iter < pyramid_height; ++iter) {
    border.x += stencil_size.x;
    border.y += stencil_size.y;
    border.z += stencil_size.z;
    int uidx = -1;
    // (x, y, z) is the location in the input of this thread.
#pragma omp parallel for private(uidx) shared(output)
    for (int z = border.z; z < input_size.z - border.z; ++z) {
      for (int y = border.y; y < input_size.y - border.y; ++y) {
        for (int x = border.x; x < input_size.x - border.x; ++x) {
          // Get current cell value or edge value.
          // uidx = ez + input_size.y * (ey * input_size.x + ex);
          uidx = x + input_size.x * (y + z * input_size.y);
          output[uidx] = OMPCellValue(input_size, x, y, z, input,
                  bornMin, bornMax, dieMin, dieMax, border);
        }
      }
    }
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

/**
 * Function exported to do the entire stencil computation.
 */
void runOMPCellOuter(DTYPE *host_data, int depth, int height, int width,
        int iterations, const int kPyramidHeight, int bornMin, int bornMax,
        int dieMin, int dieMax) {
  //printf("runOMPCellOuter(depth:%d, height:%d, width:%d)\n", depth, height,
    //      width);
  // User-specific parameters
  dim3 input_size;
  dim3 stencil_size;
  dim3 border;
  stencil_size.x = 1;
  stencil_size.y = 1;
  stencil_size.z = 1;
  border.x = kPyramidHeight * stencil_size.x;
  border.y = kPyramidHeight * stencil_size.y;
  border.z = kPyramidHeight * stencil_size.z;
#ifdef EXTRA_GHOST
  input_size.z = depth + 2 * border.x;
  input_size.y = height + 2 * border.y;
  input_size.x = width + 2 * border.z;
#else
  input_size.z = depth;
  input_size.y = height;
  input_size.x = width;
#endif


  int size = input_size.x * input_size.y * input_size.z;
  if (NULL == device_input && NULL == device_output) {
    device_output = new DTYPE[size]();
    device_input = new DTYPE[size]();
  }

  copyOuterHostData(host_data, input_size, border, device_input);
  
  // Now we can calculate the pyramid height.
  int pyramid_height = kPyramidHeight;

  // Run computation
  for (int iter = 0; iter < iterations; iter += pyramid_height) {
    if (iter + pyramid_height > iterations)
      pyramid_height = iterations - iter;
    //printf("runOMPCellKernel: ph:%d iterations:%d\n", pyramid_height, iterations);
    runOMPCellKernelOuter(input_size, stencil_size, device_input, device_output,
            pyramid_height, global_ro_data, bornMin, bornMax,
            dieMin, dieMax);
    DTYPE *temp = device_input;
    device_input = device_output;
    device_output = temp;
  }
  
  copyOuterHostData(device_input, input_size, border, host_data);
}

/**
 * Function exported to do the entire stencil computation.
 */
void runOMPCellInner(DTYPE *host_data, int x_max, int y_max, int z_max,
        int iterations, const int kPyramidHeight, int bornMin, int bornMax,
        int dieMin, int dieMax) {
  // User-specific parameters
  dim3 input_size;
  dim3 stencil_size;
  dim3 border;
  // TODO: (den4gr) when integrating with SL, stencil size will come from compiler
  stencil_size.x = 1;
  stencil_size.y = 1;
  stencil_size.z = 1;
  border.x = kPyramidHeight * stencil_size.x;
  border.y = kPyramidHeight * stencil_size.y;
  border.z = kPyramidHeight * stencil_size.z;
#ifdef EXTRA_GHOST
  input_size.x = x_max + 2 * border.x;
  input_size.y = y_max + 2 * border.y;
  input_size.z = z_max + 2 * border.z;
#else
  input_size.x = x_max;
  input_size.y = y_max;
  input_size.z = z_max;
#endif


  int size = input_size.x * input_size.y * input_size.z;
  if (NULL == device_input && NULL == device_output) {
    device_output = new DTYPE[size]();
    device_input = new DTYPE[size]();
  }

  // TODO (donnie) create copy inner/outer to save time
  copyInnerHostData(host_data, input_size, border, device_input);
 
  // Now we can calculate the pyramid height.
  int pyramid_height = kPyramidHeight;

  // Run computation
  for (int iter = 0; iter < iterations; iter += pyramid_height) {
    if (iter + pyramid_height > iterations)
      pyramid_height = iterations - iter;
    //printf("runOMPCellKernel: ph:%d iterations:%d\n", pyramid_height, iterations);
    runOMPCellKernelInner(input_size, stencil_size, device_input, device_output,
            pyramid_height, global_ro_data, bornMin, bornMax,
            dieMin, dieMax);
    DTYPE *temp = device_input;
    device_input = device_output;
    device_output = temp;
  }
 
  // TODO (donnie) create copy inner/outer to save time
  copyInnerHostData(device_input, input_size, border, host_data); 
}

void runOMPCell(DTYPE *host_data, int x_max, int y_max, int z_max,
        int iterations, const int kPyramidHeight, int bornMin, int bornMax,
        int dieMin, int dieMax) {
  // User-specific parameters
  dim3 input_size;
  dim3 stencil_size;
  dim3 border;
  stencil_size.x = 1;
  stencil_size.y = 1;
  stencil_size.z = 1;
  border.x = kPyramidHeight * stencil_size.x;
  border.y = kPyramidHeight * stencil_size.y;
  border.z = kPyramidHeight * stencil_size.z;
#ifdef EXTRA_GHOST
  input_size.x = x_max + 2 * border.x;
  input_size.y = y_max + 2 * border.y;
  input_size.z = z_max + 2 * border.z;
#else
  input_size.x = x_max;
  input_size.y = y_max;
  input_size.z = z_max;
#endif


  int size = input_size.x * input_size.y * input_size.z;
  if (NULL == device_input && NULL == device_output) {
    device_output = new DTYPE[size]();
    device_input = new DTYPE[size]();
  }
  struct timeval start, end;
  
  gettimeofday(&start, NULL);
  copyHostData(host_data, input_size, border, device_input);
  gettimeofday(&end, NULL);
  memcpy_time += secondsElapsed(start, end);
  
  // Now we can calculate the pyramid height.
  int pyramid_height = kPyramidHeight;

  // Run computation
  for (int iter = 0; iter < iterations; iter += pyramid_height) {
    if (iter + pyramid_height > iterations)
      pyramid_height = iterations - iter;
    //printf("runOMPCellKernel: ph:%d iterations:%d\n", pyramid_height, iterations);
    runOMPCellKernel(input_size, stencil_size, device_input, device_output,
            pyramid_height, global_ro_data, bornMin, bornMax,
            dieMin, dieMax);
    DTYPE *temp = device_input;
    device_input = device_output;
    device_output = temp;
  }

  gettimeofday(&start, NULL);
    copyHostData(device_input, input_size, border, host_data);
  gettimeofday(&end, NULL);
  memcpy_time += secondsElapsed(start, end);
}

void runOMPCellCleanup() {
  if (device_input != NULL && device_output != NULL) {
    delete [] device_input;
    device_input = NULL;
    delete [] device_output;
    device_output = NULL;
  }
  if (global_ro_data != NULL) {
    delete [] global_ro_data;
    global_ro_data = NULL;
  }
}

/**
 * Store unnamed data on device.
 */
void runOMPCellSetData(DTYPE *host_data, int num_elements) {
  global_ro_data = new DTYPE[num_elements];
  memcpy(static_cast<void*> (global_ro_data), static_cast<void*> (host_data),
          num_elements * sizeof (DTYPE));
}

void copyHostData(DTYPE* src_data, dim3 size, dim3 border, DTYPE* dest_data) {
  struct timeval start, stop;
  const int kNumElements = size.x * size.y * size.z;
  gettimeofday(&start, NULL);
  src_data[kNumElements - 1] = 2 * dest_data[kNumElements - 1];
  std::copy(src_data, src_data + kNumElements, dest_data);
  gettimeofday(&stop, NULL);
  memcpy_time += secondsElapsed(start, stop);
}

void copyOuterHostData(DTYPE* src_data, dim3 size, dim3 border, DTYPE* dest_data) {
  //printf("copyOuterHostData(size[%d %d %d] border[%d %d %d]\n", size.z, size.y,
    //      size.x, border.z, border.y, border.x);
  // front ghost zone
  int offset = 0;
  //printf("src: %p to %p\n", src_data, src_data + size.z * size.y * size.x);
  //printf("dest: %p to %p\n", dest_data, dest_data + size.z * size.y * size.x);
  
  DTYPE *src_start = src_data;
  DTYPE *src_stop = src_start + border.z * size.y * size.x;
  DTYPE *dest_start = dest_data;
  //printf("gz dim: %d x %d x %d\n", border.z, size.y, size.x);
  //printf("front gz: offset %d  src_start:%p dest_start:%p\n", offset, src_start,
      //    dest_start);
  std::copy(src_start, src_stop, dest_start);

  // back ghost zone
  offset = (size.z - border.z) * size.y * size.x;
  src_start = src_data + offset;
  src_stop = src_start + border.z * size.y * size.x;
  dest_start = dest_data + offset;
  //printf("back gz: offset %d  src_start:%p dest_start:%p\n", offset, src_start,
    //      dest_start);
  std::copy(src_start, src_stop, dest_start);

#ifndef SLAB
  // top
  for (int i = border.z; i < size.z - border.z; ++i) {
    offset = i * size.y * size.x + (size.y - border.y) * size.x;
    src_start = src_data + offset;
    src_stop = src_start + border.y * size.x;
    dest_start = dest_data + offset;
    std::copy(src_start, src_stop, dest_start);
  }

  // bottom
  for (int i = border.z; i < size.z - border.z; ++i) {
    offset = i * size.y * size.x;
    src_start = src_data + offset;
    src_stop = src_start + border.y * size.x;
    dest_start = dest_data + offset;
    std::copy(src_start, src_stop, dest_start);
  }

  // left
  for (int i = border.z; i < size.z - border.z; ++i) {
    for (int j = border.y; j < size.y - border.y; ++j) {
      offset = i * size.y * size.x + j * size.x;
      src_start = src_data + offset;
      src_stop = src_start + border.x;
      dest_start = dest_data + offset;
      std::copy(src_start, src_stop, dest_start);
    }
  }

  // right
  for (int i = border.z; i < size.z - border.z; ++i) {
    for (int j = border.y; j < size.y - border.y; ++j) {
      offset = i * size.y * size.x + j * size.x + size.x - border.x;
      src_start = src_data + offset;
      src_stop = src_start + border.x;
      dest_start = dest_data + offset;
      std::copy(src_start, src_stop, dest_start);
    }
  }
#endif
}

void copyInnerHostData(DTYPE* src_data, dim3 size, dim3 border, DTYPE* dest_data) {
#ifndef SLAB
  for (int i = border.z; i < size.z - border.z; ++i) {
    for (int j = border.y; j < size.y - border.y; ++j) {
      int offset = i * size.y * size.x + j * size.x + border.x;
      DTYPE* src_start = src_data + offset;
      DTYPE* src_stop = src_start + size.x - 2 * border.x;
      DTYPE* dest_start = dest_data + offset;
      std::copy(src_start, src_stop, dest_start);
    }
  }
#else
  int offset = border.z * size.y * size.x;
  DTYPE* src_start = src_data + offset;
  DTYPE* src_stop = src_start + (size.z - 2 * border.z) * size.y * size.x;
  DTYPE* dest_start = dest_data + offset;
  std::copy(src_start, src_stop, dest_start);
#endif
}