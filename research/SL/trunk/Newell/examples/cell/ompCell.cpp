// -*- Mode: C++ ; c-file-style:"stroustrup"; indent-tabs-mode:nil; -*-

#include "ompCell.h"
#include <stdio.h>
#include <cstring>
#include <stdlib.h>
#ifndef WIN32
#include <sys/time.h>
#else
#include<time.h>
#endif
#define DTYPE int

// The size of the tile is calculated at compile time by the SL processor.
// But the data array is statically sized.
// So, make these are big as they can get.
// Changed to be large enough for fermi
// (int)cube_rt(1024) = 10
#define TILE_WIDTH  10
#define TILE_HEIGHT 10
#define TILE_DEPTH  10

typedef struct dim{  int x;  int y;  int z;}dim3;

// Macro to read global read only data from within CellValue code.
#define read(offset)(ro_data[offset])

DTYPE OMPCellValue(dim3 input_size, int x, int y, int z, DTYPE *input
    , int bornMin, int bornMax, int dieMin, int dieMax)
{
  int uidx = x + input_size.x * (y + z * input_size.y);
  int orig = input[uidx];
  int sum = 0;
  int i, j, k;
  for (i = -1; i <= 1; i++)
  {  
    for (j = -1; j <= 1; j++)
    { 
      for (k = -1; k <= 1; k++)
      {  
        uidx = x+k + input_size.x * (y+j + (z+i) * input_size.y);
        sum += input[uidx];
      }
    }
  }
  sum -= orig;
  int retval;
  if(orig>0 && (sum <= dieMax || sum >= dieMin)) 
    retval = 0;
  else if (orig==0 && (sum >= bornMin && sum <= bornMax)) 
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
// This avoids name mangling and allows us to get attributes about the kernel call from Cuda.
// Its possible to do this with a C++ interface, but that will only run on certain devices.
// This technique is older and therefore more reliable across Cuda devices.
extern "C" {
  void runOMPCellKernel(dim3 input_size, dim3 stencil_size,
      DTYPE *input, DTYPE *output, int pyramid_height,
      DTYPE *ro_data
      , int bornMin, int bornMax, int dieMin, int dieMax);
}


void runOMPCellKernel(dim3 input_size, dim3 stencil_size,
    DTYPE *input, DTYPE *output, int pyramid_height,
    DTYPE *ro_data
    , int bornMin, int bornMax, int dieMin, int dieMax)
{
  dim3 border;
  int bx, by, bz, tx, ty, tz, ex, ey, ez, uidx, iter, inside=1;
  DTYPE value;

  border.x = border.y = border.z = 0;

  for(int iter = 0; iter < pyramid_height; ++iter) 
  {
    border.x += stencil_size.x;
    border.y += stencil_size.y;
    border.z += stencil_size.z;

    // (x, y, z) is the location in the input of this thread.
    for(int z = border.z; z<input_size.z-border.z; ++z)
    {
      for(int y = border.y; y<input_size.y-border.y; ++y)
      {
        for(int x = border.x; x<input_size.x-border.x; ++x)
        {
          // Get current cell value or edge value.
          // uidx = ez + input_size.y * (ey * input_size.x + ex);
          uidx = x + input_size.x * (y + z * input_size.y);

          value = OMPCellValue(input_size, x, y, z, input
              , bornMin, bornMax, dieMin, dieMax);
          output[uidx] = value;
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
static DTYPE *device_input=NULL, *device_output=NULL;
/**
 * Function exported to do the entire stencil computation.
 */
void runOMPCell(DTYPE *host_data, int x_max, int y_max, int z_max, int iterations
    , int bornMin, int bornMax, int dieMin, int dieMax, int device)
{
  // User-specific parameters
  dim3 input_size;
  input_size.x=x_max;
  input_size.y=y_max;
  input_size.z=z_max;
  dim3 stencil_size;
  stencil_size.x=1;
  stencil_size.y=1;
  stencil_size.z=1;

  //use the appropriate device
  int curr_device = -1;

  // Host to device
  int size = input_size.x * input_size.y * input_size.z;
  if(NULL==device_input && NULL==device_output)
  {
    device_output = new DTYPE[size];
    device_input = new DTYPE[size];
  }
  
  memcpy((void*)device_input, (void*)host_data, size*sizeof(DTYPE));

  dim3 border;
  
  // Now we can calculate the pyramid height.
  int pyramid_height = 1;

  // Run computation
  for (int iter = 0; iter < iterations; iter += pyramid_height)
  {
    if (iter + pyramid_height > iterations)
      pyramid_height = iterations - iter;
    runOMPCellKernel(
        input_size, stencil_size, device_input, device_output,
        pyramid_height, global_ro_data
        , bornMin, bornMax, dieMin, dieMax);
    DTYPE *temp = device_input;
    device_input = device_output;
    device_output = temp;
  }

  memcpy((void*)host_data, (void*)device_input, size*sizeof(DTYPE));

  if (global_ro_data != NULL)
  {
    delete [] global_ro_data;
    global_ro_data = NULL;
  }
}

void runOMPCellCleanup()
{
  if (device_input != NULL && device_output != NULL)
  {
    delete [] device_input;
    device_input=NULL;
    delete [] device_output;
    device_output=NULL;
  }
}

/**
 * Store unnamed data on device.
 */
void runOMPCellSetData(DTYPE *host_data, int num_elements)
{
  global_ro_data=new DTYPE[num_elements];
  memcpy((void*)global_ro_data, (void*)host_data, num_elements*sizeof(DTYPE));
}
