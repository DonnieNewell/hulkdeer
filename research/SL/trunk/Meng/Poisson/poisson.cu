#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "timer.h"
#include "blocksize.h"
//#define BLOCK_SIZE 20

#define STR_SIZE 256

#define DEVICE 1

# define HALO 1 // add one iteration will extend the pyramid base by 2 per each borderline

#define M_SEED 9

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}


#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))


void
init(int rows, int cols, float* data)
{
	float **wall = new float*[rows]; 	
	for(int n=0; n<rows; n++) 		
		wall[n]=data+cols*n; 	 	
	int seed = M_SEED; 	
	srand(seed);  	
	for (int i = 0; i < rows; i++)     {         
		for (int j = 0; j < cols; j++)         {             
			wall[i][j] = (rand() % 1000)*0.001;         
		}     
	} 
	#ifdef BENCH_PRINT     
	for (int i = 0; i < rows; i++)     {         
		for (int j = 0; j < cols; j++)         {             
			printf("%d ",wall[i][j]) ;         
		}         
		printf("\n") ;     
	} 
	#endif
}

__global__ void calculate_temp(int iteration,  //number of iteration
                               int trpzheight, 
                               float *tempIn,    //temperature input
                               float *tempOut,    //temperature output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
							   int border_cols,  // border offset 
							   int border_rows  // border offset
                               ){
	
        __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result
        
	int bx = blockIdx.x;
        int by = blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;
			
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	int small_block_rows = BLOCK_SIZE-trpzheight*HALO*2;
	int small_block_cols = BLOCK_SIZE-trpzheight*HALO*2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;

        // calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;

        // load data if it is within the valid input range
        int index = grid_rows*yidx+xidx;

        int N = ty-1;
        int S = ty+1;
        int W = tx-1;
        int E = tx+1;

        N = (N+blkY < 0 ) ? -blkY : N;
        S = (S+blkY >= grid_rows) ? grid_rows-1-blkY : S;
        W = (W+blkX < 0 ) ? -blkX : W;
        E = (E+blkX >= grid_cols) ? grid_cols-1-blkX : E;

        bool isValid = false;
	if(IN_RANGE(yidx, 0, grid_rows-1) && IN_RANGE(xidx, 0, grid_cols-1)){
            temp_on_cuda[ty][tx] = tempIn[index];  // Load the temperature data from global memory to shared memory
            isValid = true;
	}
        if(!isValid)
            return;

	__syncthreads();

        int i;
        for (i=0; i<iteration ; i++){ 
            temp_t[ty][tx] =  0.25*(temp_on_cuda[S][tx] + \
						temp_on_cuda[N][tx] + \
						temp_on_cuda[ty][E] + \
						temp_on_cuda[ty][W]) - \
						temp_on_cuda[ty][tx]*temp_on_cuda[ty][tx];
            __syncthreads();
            if(i==iteration-1)
                break;
            temp_on_cuda[ty][tx]= temp_t[ty][tx];
            __syncthreads();
          }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if( IN_RANGE(tx, trpzheight, BLOCK_SIZE-trpzheight-1) &&  \
                  IN_RANGE(ty, trpzheight, BLOCK_SIZE-trpzheight-1) &&  \
                  isValid ) {
          tempOut[index]= temp_t[ty][tx];		
      }
}

/*
   compute N time steps
*/

int compute_tran_temp(float *MatrixTemp[], int col, int row, \
		int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows) 
{
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(blockCols, blockRows);  
	
	float t;

        int src = 1;
        int dst = 0;
	for (t = 0; t < total_iterations; t+=num_iterations) {
            int temp = src;
            src = dst;
            dst = temp;
            calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations-t), \
                num_iterations,\
                MatrixTemp[src], MatrixTemp[dst],\
		col,row,borderCols, borderRows);
	}
        return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    int size;
    int grid_rows,grid_cols;
    float *FilesavingTemp, *MatrixOut;
    
    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations
    if (argc >= 2)
    {
	grid_rows = atoi(argv[1]);
	grid_cols = atoi(argv[1]);
    }
    if (argc >= 3)
        pyramid_height = atoi(argv[2]);
    if (argc >= 4)
        total_iterations = atoi(argv[3]);
    else{

	printf("Usage: hotspot grid_rows_and_cols pyramid_height iterations\n");
        exit(0);
    }

    size=grid_rows*grid_cols;

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int borderRows = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    FilesavingTemp = (float *) malloc(size*sizeof(float));
    MatrixOut = (float *) calloc (size, sizeof(float));

    if( !FilesavingTemp || !MatrixOut)
        fatal("unable to allocate memory");

    printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockSize: %d\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",\
	pyramid_height, grid_cols, grid_rows, borderCols, borderRows, BLOCK_SIZE, blockCols, blockRows, smallBlockCol, smallBlockRow);
	
    init(grid_rows, grid_cols, FilesavingTemp);

    float *MatrixTemp[2];
    cudaMalloc((void**)&MatrixTemp[0], sizeof(float)*size);
    cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(float)*size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&MatrixTemp[1], sizeof(float)*size);

    unsigned long long cycles;
    pin_stats_reset();
    int ret = compute_tran_temp(MatrixTemp,grid_cols,grid_rows, \
	 total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows);

    cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(float)*size, cudaMemcpyDeviceToHost);

    pin_stats_pause(cycles);
    pin_stats_dump(cycles);

    cudaFree(MatrixTemp[0]);
    cudaFree(MatrixTemp[1]);
    free(MatrixOut);
}
