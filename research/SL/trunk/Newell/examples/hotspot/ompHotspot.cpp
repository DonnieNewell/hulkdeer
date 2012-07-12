/*
  Copyright 2012 Donald Newell
*/

#include "ompHotspot.h"
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
#define PYRAMID_HEIGHT 1

typedef struct dim {
  int x;
  int y;
  int z;
} dim3;

/**
 * Store data between calls to SetData() and run().
 * This is basically a hack.
 */
static DTYPE *omp_global_ro_data = NULL;

/**
 * this depends on all blocks being the same size
 */
static DTYPE *device_input = NULL, *device_output = NULL;

// Macro to read read only data from within CellValue code.
#define read(offset)(omp_global_ro_data[offset])
#define get(x_off, y_off) ( ro_data[x + x_off + (y + y_off) * input_size.x] )

DTYPE CellValue(dim3 input_size, int x, int y, DTYPE *ro_data, 
                float step_div_Cap, float Rx, float Ry, float Rz, dim3 border) {
  int uidx = (x - border.x) + (y - border.y) * (input_size.x - 2 * border.x);
  float pvalue, value, term1, term2, term3, sum;
  
  pvalue = read(uidx);
  value = get(0, 0);
  term1 = (get(0, 1) + get(0, -1) - value - value) / Ry;
  term2 = (get(1, 0) + get(-1, 0) - value - value) / Rx;
  term3 = (80.0 - value) / Rz;
  sum = pvalue + term1 + term2 + term3;
  return value + step_div_Cap * sum;
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
  void runOMPHotspotKernel(dim3 input_size, dim3 stencil_size,
      DTYPE *input, DTYPE *output, int pyramid_height,
      DTYPE *ro_data, float step_div_Cap, float Rx, float Ry, float Rz);
}


void runOMPHotspotKernel(dim3 input_size, dim3 stencil_size, DTYPE *input,
                      DTYPE *output, int pyramid_height, DTYPE *ro_data,
                      float step_div_Cap, float Rx, float Ry, float Rz) {
  //fprintf(stderr, "omp_global_ro_data:%p\n", omp_global_ro_data);
  dim3 border;
  border.x = border.y = border.z = 0;
  for (int iter = 0; iter < pyramid_height; ++iter) {
    border.x += stencil_size.x;
    border.y += stencil_size.y;
    int uidx = -1;
    // (x, y, z) is the location in the input of this thread.
  #pragma omp parallel for private(uidx) shared(output)
    for (int y = border.y; y < input_size.y - border.y; ++y) {
      for (int x = border.x; x < input_size.x - border.x; ++x) {
          // Get current cell value or edge value.
          // uidx = ez + input_size.y * (ey * input_size.x + ex);
          uidx = x + input_size.x * y;

          output[uidx] = CellValue(input_size, x, y, input, step_div_Cap,
                                    Rx, Ry, Rz, border);
      }
    }
  }
}

void copyFromHostData(DTYPE* dest_data, DTYPE* host_data, dim3 size, dim3 border) {
  int length_y = size.y - 2 * border.y;
  int length_x = size.x - 2 * border.x;
  for (int i = 0; i < size.y; ++i) {
    for (int j = 0; j < size.x; ++j) {
      int src_i = i;
      int src_j = j;

      // set i
      if (src_i < border.y)
        src_i = 0;
      else if (src_i >= border.y && src_i < (size.y - border.y - 1))
        src_i -= border.y;
      else
        src_i = length_y - 1;

      // set j
      if (src_j < border.x)
        src_j = 0;
      else if (src_j >= border.x && src_j < (size.x - border.x - 1))
        src_j -= border.x;
      else
        src_j = length_x - 1;

      int src_index = src_i * length_x + src_j;
      int dest_index = i * size.x + j;
      dest_data[dest_index] = host_data[src_index];
    }
  }

}

void copyToHostData(DTYPE* host_data, DTYPE* src_data, dim3 size_src,
        dim3 border) {
  const int kLengthY = size_src.y - 2 * border.y;
  const int kLengthX = size_src.x - 2 * border.x;
  const int kOffsetY = border.y;
  const int kOffsetX = border.x;
  for (int i = 0; i < kLengthY; ++i) {
    for (int j = 0; j < kLengthX; ++j) {
      int source_index = (kOffsetY + i) * size_src.x + (kOffsetX + j);
      int destination_index = i * kLengthX + j;
      host_data[destination_index] = src_data[source_index];
    }
  }
}

/**
 * Function exported to do the entire stencil computation.
 */
void runOMPHotspot(DTYPE *host_data, int x_max, int y_max,
                int iterations, float step_div_Cap, float Rx, float Ry,
                float Rz) {
  //fprintf(stderr, "runOMPHotspot(host_data:%p, x_m:%d, y_m:%d, iterations:%d);", host_data, x_max, y_max, iterations);
  // User-specific parameters
  dim3 stencil_size;
  stencil_size.x = 1;
  stencil_size.y = 1;
  dim3 border;
  border.x = PYRAMID_HEIGHT * stencil_size.x;
  border.y = PYRAMID_HEIGHT * stencil_size.y;
  dim3 input_size;
  input_size.x = x_max + 2 * border.x;
  input_size.y = y_max + 2 * border.y;

  int size = input_size.x * input_size.y;
  if (NULL == device_input && NULL == device_output) {
    device_output = new DTYPE[size]();
    device_input = new DTYPE[size]();
  }

  copyFromHostData(device_input, host_data, input_size, border);
  
  // Now we can calculate the pyramid height.
  int pyramid_height = PYRAMID_HEIGHT;

  // Run computation
  for (int iter = 0; iter < iterations; iter += pyramid_height) {
    if (iter + pyramid_height > iterations)
      pyramid_height = iterations - iter;
    runOMPHotspotKernel(input_size, stencil_size, device_input, device_output,
                      pyramid_height, omp_global_ro_data, step_div_Cap, Rx, Ry, Rz);
    DTYPE *temp = device_input;
    device_input = device_output;
    device_output = temp;
  }

  copyToHostData(host_data, device_input, input_size, border);
}

void runOMPHotspotCleanup() {
  if (device_input != NULL && device_output != NULL) {
    delete [] device_input;
    device_input = NULL;
    delete [] device_output;
    device_output = NULL;
  }
  if (omp_global_ro_data != NULL) {
    delete [] omp_global_ro_data;
    omp_global_ro_data = NULL;
  }
}

/**
 * Store unnamed data on device.
 */
void runOMPHotspotSetData(DTYPE *host_data, int num_elements) {
  omp_global_ro_data = new DTYPE[num_elements];
  memcpy(static_cast<void*>(omp_global_ro_data), static_cast<void*>(host_data),
          num_elements*sizeof(DTYPE));
}
