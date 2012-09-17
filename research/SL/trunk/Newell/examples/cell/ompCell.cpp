/*
  Copyright 2012 Donald Newell
 */

#include "ompCell.h"
#ifndef WIN32
#include <sys/time.h>
#else
#include<time.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

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

void copyFromHostData(DTYPE* dest_data, DTYPE* host_data, dim3 size, dim3 border);
void copyToHostData(DTYPE* host_data, DTYPE* src_data, dim3 size_src,
        dim3 border);

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
      for (int y = kGhostY; y < input_size.y - kGhostY; ++y) {
        for (int x = kGhostX; x < input_size.x - kGhostX; ++x) {
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
void runOMPCellOuter(DTYPE *host_data, int x_max, int y_max, int z_max,
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
  input_size.x = x_max + 2 * border.x;
  input_size.y = y_max + 2 * border.y;
  input_size.z = z_max + 2 * border.z;


  int size = input_size.x * input_size.y * input_size.z;
  if (NULL == device_input && NULL == device_output) {
    device_output = new DTYPE[size]();
    device_input = new DTYPE[size]();
  }

  copyFromHostData(device_input, host_data, input_size, border);

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

  copyToHostData(host_data, device_input, input_size, border);
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
  stencil_size.x = 1;
  stencil_size.y = 1;
  stencil_size.z = 1;
  border.x = kPyramidHeight * stencil_size.x;
  border.y = kPyramidHeight * stencil_size.y;
  border.z = kPyramidHeight * stencil_size.z;
  input_size.x = x_max + 2 * border.x;
  input_size.y = y_max + 2 * border.y;
  input_size.z = z_max + 2 * border.z;


  int size = input_size.x * input_size.y * input_size.z;
  if (NULL == device_input && NULL == device_output) {
    device_output = new DTYPE[size]();
    device_input = new DTYPE[size]();
  }

  // TODO (donnie) create copy inner/outer to save time
  copyFromHostData(device_input, host_data, input_size, border);

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
  copyToHostData(host_data, device_input, input_size, border);
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
  input_size.x = x_max + 2 * border.x;
  input_size.y = y_max + 2 * border.y;
  input_size.z = z_max + 2 * border.z;


  int size = input_size.x * input_size.y * input_size.z;
  if (NULL == device_input && NULL == device_output) {
    device_output = new DTYPE[size]();
    device_input = new DTYPE[size]();
  }

  copyFromHostData(device_input, host_data, input_size, border);

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

  copyToHostData(host_data, device_input, input_size, border);
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

void copyFromHostData(DTYPE* dest_data, DTYPE* host_data, dim3 size, dim3 border) {
  int length_y = size.y - 2 * border.y;
  int length_x = size.x - 2 * border.x;
  int length_z = size.z - 2 * border.z;
  for (int i = 0; i < size.z; ++i) {
    for (int j = 0; j < size.y; ++j) {
      for (int k = 0; k < size.x; ++k) {
        int src_i = i;
        int src_j = j;
        int src_k = k;

        // set i
        if (src_i < border.z)
          src_i = 0;
        else if (src_i >= border.z && src_i < (size.z - border.z - 1))
          src_i -= border.z;
        else
          src_i = length_z - 1;

        // set j
        if (src_j < border.y)
          src_j = 0;
        else if (src_j >= border.y && src_j < (size.y - border.y - 1))
          src_j -= border.y;
        else
          src_j = length_y - 1;

        // set k
        if (src_k < border.x)
          src_k = 0;
        else if (src_k >= border.x && src_k < (size.x - border.x - 1))
          src_k -= border.x;
        else
          src_k = length_x - 1;

        int src_index = (src_i * length_y + src_j) * length_x + src_k;
        int dest_index = (i * size.y + j) * size.x + k;
        dest_data[dest_index] = host_data[src_index];
      }
    }
  }
}

void copyToHostData(DTYPE* host_data, DTYPE* src_data, dim3 size_src,
        dim3 border) {
  const int kLengthZ = size_src.z - 2 * border.z;
  const int kLengthY = size_src.y - 2 * border.y;
  const int kLengthX = size_src.x - 2 * border.x;
  const int kOffsetZ = border.z;
  const int kOffsetY = border.y;
  const int kOffsetX = border.x;
  for (int i = 0; i < kLengthZ; ++i) {
    for (int j = 0; j < kLengthY; ++j) {
      for (int k = 0; k < kLengthX; ++k) {
        int source_index = ((kOffsetZ + i) * size_src.y + (kOffsetY + j)) *
                size_src.x + (kOffsetX + k);
        int destination_index = (i * kLengthY + j) * kLengthX + k;
        host_data[destination_index] = src_data[source_index];
      }
    }
  }
}
