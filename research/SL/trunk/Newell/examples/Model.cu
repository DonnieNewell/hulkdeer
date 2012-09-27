/* -*- Mode: C ; indent-tabs-mode: nil ; c-file-style: "stroustrup" -*-

    CS 6620 - Compilers
    Stencil App Language Project
    Authors: Greg Faust, Sal Valente

    File:   Model.cu     Contains the analytical model for predicting stencil app latencies based on input sizes and trapezoid height.

    TODO    For now, this is more or less a direct translation of the MatLab code.
            All of that code assumes all the data and blocks are the same size in all dimensions.
            Once we get this working, we might consider relaxing that assumption.
            Also, MatLab is not typed.  
            Apparently, the current latencies are in terms of GPU cycles.
            To avoid step functions, I have used doubles throughout.
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>
#ifndef WIN32
	#include <sys/time.h>
#else
	#include <time.h>
#endif
#include "Model.h"

// This turns on the debugging capabilities of things defined in cutil.h
#define _DEBUG

// First, give some simple math helper functions.
double iexp(double base, int power) {
    double retval = 1;
    for (int i = 0; i < power; ++i) retval *= base;
    return retval;
} 

// This will return the largest int that goes into composite, possibly with leftover.
// That is, it is floor of the real log.
int ilog(int root, int composite)
{
    double retd = pow(composite, 1.0/root);
    // Give it a little wiggle room for floating point errors.
    return (int)(retd + .02);
}

// Some CUDA helper furnctions.

inline dim3 filldim3(dim3 * d3, int x = 1, int y = 1, int z = 1)
{
    d3->x = x;
    d3->y = y;
    d3->z = z;
    return *d3;
}

inline dim3 copydim3(dim3 in, dim3 out)
{
    out.x = in.x;
    out.y = in.y;
    out.z = in.z;
    return out;
}

// A debug helper.
// We can probably get rid of this before code freeze.
int printCudaDevices()
{
    int curDev;
    CUDA_SAFE_CALL(cudaGetDevice(&curDev));
    fprintf(stderr, "Current cuda device is: %d.\n", curDev);
    int devCount;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&devCount));
    fprintf(stderr, "There are %d cuda devices on this machine.\n", devCount);
    int i;
    CudaDeviceProps_t * devProps = (CudaDeviceProps_t *)malloc(sizeof(CudaDeviceProps_t));
    for (i=0; i<devCount; i++)
    {
        CUDA_SAFE_CALL(cudaGetDeviceProperties(devProps, i));
        printCudaDeviceProps(devProps);
    }
    free(devProps);
    return curDev;
}

// A debug helper.
// We can probably get rid of this before code freeze.
void printCudaDeviceProps (CudaDeviceProps_t * devProps)
{
    fprintf(stdout, "CUDA device \"%s\" properites.\n", devProps->name);
    fprintf(stdout, "Release %d.%d.\n", devProps->major, devProps->minor);
    // fprintf(stdout, "Compute Mode=%d.\n", devProps->computeMode);
    fprintf(stdout, "Global Memory Size=%zd.\n", devProps->totalGlobalMem);
    fprintf(stdout, "Shared Memory Per Block=%zd.\n", devProps->sharedMemPerBlock);
    fprintf(stdout, "Registers Per Block=%d.\n", devProps->regsPerBlock);
    fprintf(stdout, "Clock Rate (KiloHertz)=%d.\n", devProps->clockRate);
    fprintf(stdout, "Warp Size=%d.\n", devProps->warpSize);
    fprintf(stdout, "Maximum Threads per Block=%d.\n", devProps->maxThreadsPerBlock);
    fprintf(stdout, "Maximum Block Dimensions=[%d, %d, %d].\n", devProps->maxThreadsDim[0], devProps->maxThreadsDim[1], devProps->maxThreadsDim[2]);
    fprintf(stdout, "Maximum Grid Dimensions=[%d, %d, %d].\n", devProps->maxGridSize[0], devProps->maxGridSize[1], devProps->maxGridSize[2]);
}

// We need these as part of the runtime system.
CudaDeviceProps_t * getCudaDeviceProps (int devNum)
{
    CudaDeviceProps_t * devProps = (CudaDeviceProps_t *)malloc(sizeof(CudaDeviceProps_t));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(devProps, devNum));
    return devProps;
}

CudaDeviceProps_t * getCurrentCudaDeviceProps ()
{
    int curDev;
    CUDA_SAFE_CALL(cudaGetDevice(&curDev));
    return getCudaDeviceProps(curDev);
}

// We need this as part of the runtime system.
CudaFunctionAtts_t * getCudaFunctionAtts (char * functionName)
{
    CudaFunctionAtts_t * FAs = (CudaFunctionAtts_t * )malloc(sizeof(CudaFunctionAtts_t));
    CUDA_SAFE_CALL(cudaFuncGetAttributes(FAs, "groupClumps"));    // "_Z11groupClumpssiiPtPjS0_p8fragment"));
    // fprintf(stderr, "Max Threads per block in groupClumps=%d and register count=%d\n", FAs->maxThreadsPerBlock, FAs->numRegs);
    return FAs;
}

// Make a SACuda Latency structure, and give default values to all fields.
SACudaLats_t * makeSACudaLats()
{
    SACudaLats_t * SACLs = (SACudaLats_t *)(malloc(sizeof(SACudaLats_t)));
    SACLs->totalLat = 0;
    SACLs->avgLoadLat = 0;
    SACLs->avgStoreLat = 0;
    SACLs->avgCompLat = 0;
    SACLs->avgSetupCompLat = 0;
    SACLs->avgEmbedMemLat = 0;
    SACLs->avgGlobalSyncLat = 0;
    return SACLs;
};

// Dispose of a SACuda Latency structure.
void disposeSACudaLats(SACudaLats_t * SACLs)
{
    free(SACLs);
    SACLs=NULL;
}

// Print out SACL values to aid in debugging.
void printSACLs(SACudaLats_t * SACLs, int pyramidH)
{
    fprintf(stderr, "SACL avgLoadLat=%f, total=%f\n", SACLs->avgLoadLat, SACLs->avgLoadLat * pyramidH);
    fprintf(stderr, "SACL avgStoreLat=%f, total=%f\n", SACLs->avgStoreLat, SACLs->avgStoreLat * pyramidH);
    fprintf(stderr, "SACL avgEmbedMemLat=%f, total=%f\n", SACLs->avgEmbedMemLat, SACLs->avgEmbedMemLat * pyramidH);
    fprintf(stderr, "SACL avgCompLat=%f, total=%f\n", SACLs->avgCompLat, SACLs->avgCompLat * pyramidH);
    fprintf(stderr, "SACL avgSetupCompLat=%f, total=%f\n", SACLs->avgSetupCompLat, SACLs->avgSetupCompLat * pyramidH);
    fprintf(stderr, "SACL avgGlobalSynchLat=%f, total=%f\n", SACLs->avgGlobalSyncLat, SACLs->avgGlobalSyncLat * pyramidH);
    fprintf(stderr, "SACL TotalLat=%f, total=%f\n", SACLs->totalLat, SACLs->totalLat * pyramidH);
}


static SAProps_t * SAPs=NULL;

// Make a SAProps structure, and give default values to some fields.
dim3 initSAProps(int dims, dim3 input_size, dim3 stencil_size, int iterations, int dataESize, char * kernelName)
{
    SAPs = (SAProps_t *)(malloc(sizeof(SAProps_t)));
    
    // These all come directly from the input args to the app or in the stencil language.
    SAPs->numDims = dims;
    SAPs->dataDims = input_size;
    SAPs->haloDims = stencil_size;
    SAPs->iterations = iterations;
    SAPs->dataElemSize = dataESize;

    // TODO In order to get these right, we will need a new stencil app declaration of data reads in the CellValue calculation.
    // This used to be 1 for all 4 samples except hotSpot (where it was 2).
    // Now it is 1 for ALL apps.
    SAPs->setupGlobalLoadsPerCell = 1;
    // This used to be 0 for all 4 samples except pathfinder (where it was 1).
    // But now it will be 1 for any app that accesses global read only data.
    // The model does not seem very sensitive to this value.
    SAPs->loopGlobalLoadsPerCell = 1;

    // This is 1 for all 4 samples except cell (where it is 2 because of the dimensionality.
    // I see no way to derive this value from the app at all.
    SAPs->bankConflict = (dims < 3) ? 1 : 2;

    // Now we will calculate the block dimensions.
    // Since the app will ALWAYS run faster with larger block size,
    // We will make the blocks as big as they can be on the device.
    // TODO We could perhaps also look at the shared memory and make sure we fit.
    // But for now, this is not a limiting factor for any of the sample apps.
    CudaFunctionAtts_t * CFAs = (CudaFunctionAtts_t *)malloc(sizeof(CudaFunctionAtts_t));
    SAPs->CFAs = CFAs;
    //CUDA_SAFE_CALL(cudaFuncGetAttributes(CFAs, kernelName));
    cudaFuncGetAttributes(CFAs, kernelName);
    //fprintf(stderr, "Max Threads per block in %s=%d, register count=%d, sharedMemUsage=%d.\n", kernelName, CFAs->maxThreadsPerBlock, CFAs->numRegs, CFAs->sharedSizeBytes);
    int blockLen = ilog(dims, CFAs->maxThreadsPerBlock);
    // fprintf(stderr, "Block Length=%d.\n", blockLen);
    // The block size can't be larger than the data size!
    SAPs->blockDims.x = MIN(blockLen, static_cast<int>(SAPs->dataDims.x));
    SAPs->blockDims.y = SAPs->blockDims.z = 1;
    if (dims > 1) SAPs->blockDims.y = MIN(blockLen, static_cast<int>(SAPs->dataDims.y));
    if (dims > 2) SAPs->blockDims.z = MIN(blockLen, static_cast<int>(SAPs->dataDims.z));

    // Fill in the cuda device properties.
    SAPs->CDPs = getCurrentCudaDeviceProps ();

    return SAPs->blockDims;
}

void disposeSAProps(SAProps_t * SAPs) {
  if (NULL != SAPs) {
    if (NULL != SAPs->CFAs) {
      free(SAPs->CFAs);
      SAPs->CFAs=NULL;
    }
    if (NULL != SAPs->SACLs) {
      free(SAPs->SACLs);
      SAPs->SACLs=NULL;
    }
    if (NULL != SAPs->CDPs) {
      free(SAPs->CDPs);
      SAPs->CDPs=NULL;
    }
    free(SAPs);
    SAPs = NULL;
  }
}

// TODO It would be better if this were not a macro.
// However, it calls kernel functions.
// To make it a function, we would have to figure out how to get 
//     function ptrs for kernels, and with names generated by the stencil tool.
// NOTE: I tried calling the kernel 3 times and averaging, but it did not materially improve things.
#ifndef WIN32
static struct timeval starttime, endtime;
#else
//define windows timing stuff here
clock_t startclock, endclock;
#endif
static unsigned int usec;
#ifndef WIN32
#define timeInMicroSeconds(var, funcall)                                  \
    ({                                                                    \
    gettimeofday(&starttime, NULL);                                       \
    funcall;                                                              \
    CUDA_SAFE_THREAD_SYNC();                                              \
    funcall;                                                              \
    CUDA_SAFE_THREAD_SYNC();                                              \
    funcall;                                                              \
    CUDA_SAFE_THREAD_SYNC();                                              \
    gettimeofday(&endtime, NULL);                                         \
    usec = ((endtime.tv_sec - starttime.tv_sec) * 1000000 +               \
            (endtime.tv_usec - starttime.tv_usec));                       \
    var = usec/3;                                                           \
    })
#else
#define timeInMicroSeconds(var, funcall)                                  \
    ({                                                                    \
    startclock = clock();                                       \
    funcall;                                                              \
    CUDA_SAFE_THREAD_SYNC();                                              \
    funcall;                                                              \
    CUDA_SAFE_THREAD_SYNC();                                              \
    funcall;                                                              \
    CUDA_SAFE_THREAD_SYNC();                                              \
    endclock = clock()                                         \
    usec = ((endclock - startclock) * 1000000 / CLOCKS_PER_SEC  ;             \
    var = usec/3;                                                           \
    })
#endif
// An empty kernel used to measure kernel call overhead.
// Note that how long this takes to run depends a lot on the size of block and grid.
__global__ void dummyKernel ()
{
}

//////////////////////////////////////////////////////////////////////////////////
// Start of the translation of Jiayuan Meng's MatLab code.
//////////////////////////////////////////////////////////////////////////////////

// Some helper functions for the main routine.
static inline int div_ceil(int num, int denom) {
    return (int)((num + denom - 1) / denom);
}

// These are called by the model.
// Most could probably be inlined.
double workingSet(int edge, int dimension) {
    return (double)(iexp(edge,dimension));
}

double memLat(double numElements, int coalesceWidth, double memQueueLat,
        double uncontendedLat) {
    double concurrentRequests = ((double)numElements)/coalesceWidth;
    return (concurrentRequests*memQueueLat) + uncontendedLat;
}

double pyramidMemLat(int edge, int numBlocks, int halo, int dimension,
        int pyramidHeight, int coalesceWidth, double memQueueLat,
        double uncontendedLat) {
    double set = workingSet(edge-halo, dimension)*numBlocks;
    return pyramidHeight*memLat(set, coalesceWidth, memQueueLat, uncontendedLat);
}

double blockCompLat(double numElements, double IPC,
        double instPerElementPerWarp) {
    return ((double)instPerElementPerWarp)/IPC*numElements;
}

double pyramidBlockCompLat(int edge, int halo, int dimension, int pyramidHeight,
        double IPC, double instPerElementPerWarp) {
    double set = workingSet(edge-halo, dimension);
    return pyramidHeight*blockCompLat(set, IPC, instPerElementPerWarp);
}

// This is the main workhorse routine for the model.
// It takes in properties of the Stencil Application, the Cuda device the app
// will run on, and the block and trapezoid sizes.
// From these it calculates all the predicted latencies.
// This should be called (repeatedly) by some routine that does the optimization
// of the block side and trapezoid height.
double calcSACudaLats(SAProps_t * SAProps, int blockSize, int pyramidHeight) {
    CudaDeviceProps_t * CDPs = SAPs->CDPs;
    SACudaLats_t * SACLs = SAPs->SACLs;

    // Jiayuan assumed data set was the same size in all dimensions.
    // TODO make it so that the different dimensions can be different sizes.
    double dataEdge = SAProps->dataDims.x;
    double halo = SAProps->haloDims.x;
    int dims = SAProps->numDims;
    // double numBlocks = iexp(div_ceil(dataEdge, (blockSize - (pyramidHeight*halo))), dims);    
    double numBlocks = iexp(dataEdge / (blockSize - (pyramidHeight * halo)),
                            dims);

    // This seems to be two magic constants 0.5 and bankConflict.
    double IPC = 0.5 / SAProps->bankConflict;

    // Jiayuan's comments.
    // This is for the GTX 280.
    // double glbSync = 9033.0 * CPUclock/GPUclock;
    // This is for the 9800
    // glbSync = 95847*CPUclock/GPUclock;

    // Now we get it from the device!
    double glbSync = SAProps->globalSynchLat;

    // Can we get this from the device information?
    double coalesceWidth = 16;
    // Another magic constant.
    double uncontendedLat = 300;

    // Why can't this be calculated?
    // Something like numBlocks/SAPs->CDPs->multiProcessorCount??
    // Instead, it is the number of ACTIVE blocks per MP.
    // Capped at 8 for all current Cuda Devices per Cuda programming guide.
    double numBlocksPerMP = 8;

    // Another magic constant?
    double factor = iexp(5,(dims-1));

    // Staight from MatLab.
    double requestSize = 4;
    double bandwidth_BperCycle = 141.7 / 1.3;
    double memQueueLat =
                    requestSize * coalesceWidth / bandwidth_BperCycle * factor;

    double numMPs = CDPs->multiProcessorCount;
    double numConcurrentBlocks = numBlocksPerMP*numMPs;

    // GGF If the store latency relies on the concurrency, why not the load?
    // But making this change breaks the model.
    // GGF double loadLat = ((double)numBlocks)/numConcurrentBlocks*SAProps->setupGlobalLoadsPerCell * 
    //            memLat(workingSet(blockSize, dims) * numConcurrentBlocks, coalesceWidth, memQueueLat, uncontendedLat);
    // Below is the original Jiayuan calculation.
    double loadLat = SAProps->setupGlobalLoadsPerCell *
                        memLat(workingSet(blockSize, dims) *
                        numConcurrentBlocks,
                        coalesceWidth,
                        memQueueLat,
                        uncontendedLat);
    
    // GGF Why is the calculation of the store latency so different from the load latency??
    // GGF double storeLat = memLat(workingSet(blockSize - (pyramidHeight * halo), dims) * numConcurrentBlocks, coalesceWidth, memQueueLat, uncontendedLat);
    // Below is the original Jiayuan calculation.
    double storeLat = ((double)numBlocks) / numConcurrentBlocks *
            memLat(workingSet(blockSize - (pyramidHeight * halo), dims) *
                numConcurrentBlocks,
            coalesceWidth,
            memQueueLat,
            uncontendedLat);

    double embeddedMemLat = ((double)SAProps->loopGlobalLoadsPerCell)*numBlocks / numConcurrentBlocks *
        pyramidMemLat(blockSize, numConcurrentBlocks, halo, dims, pyramidHeight, coalesceWidth, memQueueLat, uncontendedLat);

    // These values are now sampled dynamically.
    double setupInstrPerWarp = SAPs->setupInstrPerWarp;
    double totalInstrPerWarp = SAPs->totalInstrPerWarp;

    // All the below directly from MatLab, with Jiayuan's comment at line end.
    double computeLat = pyramidBlockCompLat(blockSize, halo, dims, pyramidHeight, IPC, totalInstrPerWarp - setupInstrPerWarp);
    double setupCompLat = blockCompLat(workingSet(blockSize, dims), IPC, setupInstrPerWarp); // - loadLat;
    SACLs->avgLoadLat = loadLat/pyramidHeight;                                 // going down then suddenly high [Category A, major]
    SACLs->avgStoreLat = storeLat/pyramidHeight;                               // going down then suddenly high [A, minor]
    SACLs->avgCompLat = (computeLat*(numBlocks/numMPs))/pyramidHeight;         // going higher always [Category B, major]
    SACLs->avgSetupCompLat = (setupCompLat*(numBlocks/numMPs))/pyramidHeight;  // going down then suddenly high [A, negligible]
    SACLs->avgEmbedMemLat = embeddedMemLat/pyramidHeight;                      // going higher always [B, minor]
    SACLs->avgGlobalSyncLat = glbSync/pyramidHeight;
    
    // GGF Why is the computeLat and setupCompLat multipled by the numBlocks/numMps TWICE!?!
    // GGF However, changing to below line makes the model calculate pyramid hieghts that are too high.
    // GGF SACLs->totalLat = (glbSync +  (computeLat + setupCompLat) + (loadLat + storeLat + embeddedMemLat)) / pyramidHeight;
    // Below is original Jiayuan calculation.
    SACLs->totalLat = (glbSync + (numBlocks / numMPs) * (computeLat + setupCompLat) + (loadLat + storeLat + embeddedMemLat)) / pyramidHeight;
    return SACLs->totalLat;
}

// Put all the values in an array so we can find the second best as well as the best.
// This is a real hack to avoid returning a list to the template, or having to sort.
static double minLats[1024];
static int    validLats;

int getSecond(int first)
{
    int retval = 1;
    double minLat = 1e40;
    for (int i = 1; i<=validLats; i++)
    {
        if (i == first) continue;
        if (minLats[i] < minLat)
        {
            retval = i;
            minLat = minLats[i];
        }
    }
    return retval;
}

int calcMinLatencyInternal(SAProps_t * SAPs)
{
    // Make a structure to catch the various latencies.
    SAPs->SACLs = makeSACudaLats();

    // The blocksize has already been determined.
    int blockSize = SAPs->blockDims.x;

    // initialize min to hugely big number.
    double minLat = 1e40;
    // Try all the possible pyramid heights.
    // In theory there is a descent that can be followed to a valley.
    // But sometimes there is noise near the bottom of the value.
    // And a descent will get sub-optimal value.
    // Therefore, it currently calculates all and gets minimum.
    // It is a trade-off between calculation time, and optimal running time.
    validLats = blockSize/2 - SAPs->haloDims.x;
    // fprintf(stderr, "Valid PH=%d.\n", validLats);
    for (int i=1; i<=validLats; i++)
    {
        double pyrUp = calcSACudaLats(SAPs, blockSize, i);

#ifdef STATISTICS
        fprintf(stderr, "Model Pyramid Size=%d, Model Latency=%f\n", i, pyrUp);
#endif

        // Store results to support hack to find second best.
        minLats[i] = pyrUp;

        // Model should not generate negative values.
        // So, this is a safety check.
        if (pyrUp < 0) 
        {
            break;
        }
        // Remember minimum latency, and associated pyramid height.
        if (pyrUp < minLat)
        {
            minLat = pyrUp;
            SAPs->pyramidHeight = i;
        }
        // Reinstate below line to just do the descent to the valley floor.
        // else break;
    }

    // fprintf(stderr, "BlockSize=%d, Pyramid=%d, Latency==%f\n", blockSize, SAPs->pyramidHeight, minLat);
        
    return SAPs->pyramidHeight;
}

int calcPyramidHeight(dim3 grid_dims, unsigned int oneIterTime, unsigned int twoIterTime)
{
    
    // Now we have enough information for the training period.
    // First call a kernel to make sure device is fully initialized.
    dummyKernel<<<grid_dims, SAPs->blockDims>>>();
    long int ekTime = 0;
    timeInMicroSeconds(ekTime, (dummyKernel<<<grid_dims, SAPs->blockDims>>>()));
    // fprintf(stderr, "Empty Kernel time=%u.\n", ekTime);
    // Convert micro-seconds into GPU cycles.
    if (SAPs->CDPs->major >= 2) SAPs->globalSynchLat = ((double)ekTime)*SAPs->CDPs->clockRate/1000;
    else SAPs->globalSynchLat = 3350;
    // Try setting it to Jiayuna's magic constant.
    // SAPs->globalSynchLat = 3350;

    // Try adding a bit to the synch time.
    ekTime *= 2.0;

    // fprintf(stderr, "One iter time=%u, Two iter time=%u.\n", oneIterTime, twoIterTime);

    // Now let's calculate the various latencies.
    long int oneIter =  oneIterTime;
    long int twoIter =  twoIterTime;

    // Remove the kernel call overhead.
    oneIter -= ekTime;
    twoIter -= ekTime;

    // This is the rational calculation of the times for one iteration
    //    and for the setup time.
    double fullTime = oneIter;
    double setupTime = fullTime - (twoIter - oneIter);

    // However, rational calculation does not work for small data sizes.
    // THIS IS A TOTAL HACK!
    // For some reason, we can't get good timings when things are small.
    // if (setupTime <= ekTime) setupTime = 2 * ekTime;
    // if (fullTime <= setupTime) fullTime = 2 * setupTime;
    double magic = 1.25;
    // if      (SAPs->CDPs->major >= 2) magic = 3.0;
    // else 
    if (SAPs->numDims == 2) magic = 3.0;
    if (setupTime <= 0 || fullTime < magic * ekTime)
    {   
        setupTime = MAX(setupTime, 1. * ekTime);
        fullTime = MAX(fullTime, magic * setupTime);
        //fprintf(stderr, "changed setup and full.\n");
    }
    
    // Let's use the model to calculate the best height.
    dim3 * dDims = &(SAPs->dataDims);
    dim3 * bDims = &(SAPs->blockDims);

    int WarpsPerBlock = div_ceil((bDims->x * bDims->y * bDims->z), SAPs->CDPs->warpSize);
    int numOfWarps;
    int numOfBlocks;
    // setup was run with pyramid height of 1.
    // Use div_ceil to match the grid sizes as calculated before the actual  run was made.
    numOfBlocks = iexp(div_ceil(dDims->x, bDims->x - 2*SAPs->haloDims.x), SAPs->numDims);
    numOfWarps = numOfBlocks * WarpsPerBlock;
    SAPs->setupInstrPerWarp = (((double)setupTime)*SAPs->CDPs->clockRate/1000)/numOfWarps;
    // total was run with pyramid height of 2.
    // But this makes the model worse for small data sizes.
    // numOfBlocks = div_ceil(dDims->x, bDims->x - 4*SAPs->haloDims.x) * div_ceil(dDims->y, bDims->y - 4*SAPs->haloDims.y);
    numOfWarps = numOfBlocks * WarpsPerBlock;
    SAPs->totalInstrPerWarp = (((double)fullTime)*SAPs->CDPs->clockRate/1000)/numOfWarps;

    // fprintf(stderr, "Total instructions per warp=%f.\n", SAPs->totalInstrPerWarp);
    // fprintf(stderr, "Setup instructions per warp=%f.\n", SAPs->setupInstrPerWarp);
    
    int ph = calcMinLatencyInternal(SAPs);
    return ph;
}
