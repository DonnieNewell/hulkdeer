/* -*- Mode: C ; indent-tabs-mode: nil ; c-file-style: "stroustrup" -*-

    CS 6620 - Compilers
    Stencil App Language Project
    Authors: Greg Faust, Sal Valente

    File:   Model.h     Header file for the analytical Model

*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// This flag controls whether the templates and mains are gathering statistics or running normally.
//#define STATISTICS

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

// Standard Hack to tell C++ to leave my C function names alone!!!
#ifdef __cplusplus
extern "C" {
#endif

typedef struct cudaDeviceProp CudaDeviceProps_t;
typedef struct cudaFuncAttributes CudaFunctionAtts_t;

int printCudaDevices();
void printCudaDeviceProps (CudaDeviceProps_t * devProps);
CudaDeviceProps_t * getCudaDeviceProps(int devNum); 
CudaDeviceProps_t * getCurrentCudaDeviceProps ();

typedef uint8_t BOOL;
#define TRUE (1)
#define FALSE (0)

// This structure holds all of the values calculated by the model.
// Of course, the most important is the totalLat (total latency).
// All latencies are in terms of nanoseconds (TODO Is this right?).

struct SACudaLats
{
    double totalLat;
    double avgLoadLat;
    double avgStoreLat;
    double avgCompLat;
    double avgSetupCompLat;
    double avgEmbedMemLat;
    double avgGlobalSyncLat;
};
typedef struct SACudaLats SACudaLats_t;

SACudaLats_t * makeSACudaLats();
void disposeSACudaLats(SACudaLats_t * SACLs);

// This struct will hold all the properties for a Stencil Language Application.
// Some of these values will come from the SL input.
// Others will come at runtime (in particular, the sizes).
// Others, are numbers that require app profile information.
struct SAProps
{
    // These 4 should be easy to get from the SL description, or  runtime size information.
    int           numDims;
    dim3          dataDims;
    dim3          haloDims;
    int           dataElemSize;
    int           iterations;
    // Let's hope these can come from our strategy for generating code for the SL.
    int           setupGlobalLoadsPerCell;  // Should be 2 if there is "un-named data" and 1 otherwise.
    int           loopGlobalLoadsPerCell;   // Should be 0 if there is no data loaded in the inner loop from global memory.
    // How will we get these next two?  Jiayuan got them from CudaProf on actual code runs.
    double        totalInstrPerWarp;
    double        setupInstrPerWarp;
    // This seeems it is some magic number that is 1 except for Cell it is 2.
    // The claim is that the model is not very sensitive to this number.
    // We will need to do some testing to tell.
    // If we can't get it any other way, we can make it = (numDims == 3) ? 2 : 1;
    // As the theory is that the conflict is caused by accessing data in the unusual striping of 3 dims.
    double        bankConflict;
    // These we will calculate using the model.
    dim3          blockDims;
    int           pyramidHeight;

    // We will include the other structures here so there need only be one struct visible to the template code.
    CudaFunctionAtts_t * CFAs;
    CudaDeviceProps_t * CDPs;
    SACudaLats_t * SACLs;
    // We will need some extra information about the device that is not included in the above.
    // The latency of an empty kernel call.
    double        globalSynchLat;
};
typedef struct SAProps SAProps_t;

SAProps_t * makeSAProps();
void disposeSAProps(SAProps_t * SAPs);
BOOL calcMinLatency(SAProps_t * SAPs, CudaDeviceProps_t * CDPs);

// End the cplusplus "C" extern declaration.
#ifdef __cplusplus
}
#endif
