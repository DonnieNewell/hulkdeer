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
static DTYPE *global_ro_data = NULL;

/**
 * this depends on all blocks being the same size
 */
static DTYPE *device_input = NULL, *device_output = NULL;

// Macro to read read only data from within CellValue code.
#define read(offset)(ro_data[offset])
#define get(x_off, y_off)(global_ro_data[x + x_off + (y + y_off) * input_size.x])

DTYPE CellValue(dim3 input_size, int x, int y, DTYPE *ro_data,
                float step_div_Cap, float Rx, float Ry, float Rz) {
  int uidx = x + y * input_size.x;
  float pvalue, value, term1, term2, term3, sum;
  pvalue = read(uidx);
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
  dim3 border;
  border.x = border.y = border.z = 0;
  for (int iter = 0; iter < pyramid_height; ++iter) {
    border.x += stencil_size.x;
    border.y += stencil_size.y;
    int uidx = -1;
    // (x, y, z) is the location in the input of this thread.
  #pragma omp parallel for private(uidx) shared(output)
    for (int y = border.y; y < input_size.y-border.y; ++y) {
      for (int x = border.x; x < input_size.x-border.x; ++x) {
          // Get current cell value or edge value.
          // uidx = ez + input_size.y * (ey * input_size.x + ex);
          uidx = x + input_size.x * y;
          output[uidx] = CellValue(input_size, x, y, input,
                                      step_div_Cap, Rx, Ry, Rz);
    
      }
    }
  }
}

/**
 * Function exported to do the entire stencil computation.
 */
void runOMPHotspot(DTYPE *host_data, int x_max, int y_max,
                int iterations, float step_div_Cap, float Rx, float Ry,
                float Rz) {
  // User-specific parameters
  dim3 input_size;
  input_size.x = x_max;
  input_size.y = y_max;
  dim3 stencil_size;
  stencil_size.x = 1;
  stencil_size.y = 1;
  
  int size = input_size.x * input_size.y;
  if (NULL == device_input && NULL == device_output) {
    device_output = new DTYPE[size]();
    device_input = new DTYPE[size]();
  }

  memcpy(static_cast<void*>(device_input), static_cast<void*>(host_data),
          size * sizeof(DTYPE));

  // Now we can calculate the pyramid height.
  int pyramid_height = PYRAMID_HEIGHT;

  // Run computation
  for (int iter = 0; iter < iterations; iter += pyramid_height) {
    if (iter + pyramid_height > iterations)
      pyramid_height = iterations - iter;
    runOMPHotspotKernel(input_size, stencil_size, device_input, device_output,
                      pyramid_height, global_ro_data, step_div_Cap, Rx, Ry, Rz);
    DTYPE *temp = device_input;
    device_input = device_output;
    device_output = temp;
  }

  memcpy(static_cast<void*>(host_data), static_cast<void*>(device_input),
          size*sizeof(DTYPE));

  if (global_ro_data != NULL) {
    delete [] global_ro_data;
    global_ro_data = NULL;
  }
}

void runOMPHotspotCleanup() {
  if (device_input != NULL && device_output != NULL) {
    delete [] device_input;
    device_input = NULL;
    delete [] device_output;
    device_output = NULL;
  }
}

/**
 * Store unnamed data on device.
 */
void runOMPHotspotSetData(DTYPE *host_data, int num_elements) {
  global_ro_data = new DTYPE[num_elements];
  memcpy(static_cast<void*>(global_ro_data), static_cast<void*>(host_data),
          num_elements*sizeof(DTYPE));
}
