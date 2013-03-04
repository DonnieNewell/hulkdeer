#include <stdio.h>
#include <float.h>
#include <sys/time.h>

// The number of threads per blocks in the kernel
// (if we define it here, then we can use its value in the kernel,
//  for example to statically declare an array in shared memory)
const int threads_per_block = 256;


// Forward function declarations
float *GPU_vector_max(float *A, int N);
float *CPU_add_vectors(float *A, float *B, int N);
float *get_random_vector(int N);
float cpu_max(float *A, int N);
float cpu_min(float *A, int N);
float cpu_mean(float *A, int N);
float cpu_stdev(float *A, int N);
long long start_timer();
long long stop_timer(long long start_time, char *name);
void die(char *message);


int main(int argc, char **argv) {
	// Seed the random generator (use a constant here for repeatable results)
	srand(10);

	// Determine the vector length
	int N = 100000;  // default value
	if (argc > 1) N = atoi(argv[1]); // user-specified value

	// Start the timer
	long long vector_start_time = start_timer();
        
        //create a random vector
        float *A;
        A = get_random_vector(N);
        
        stop_timer(vector_start_time, "Vector generation");
	
        long long cpu_start_time = start_timer();

        //Compute the max, min, mean, and stdev value of the vector
	float max_val = cpu_max(A,N);
	float min_val = cpu_min(A,N);
	float mean_val = cpu_mean(A,N);
	float stdev_val = cpu_stdev(A,N);

        long long CPU_time = stop_timer(cpu_start_time, "CPU");

        
        // Compute the results on the GPU
        long long GPU_start_time = start_timer();
	float* result = GPU_vector_max(A, N);
        long long GPU_time = stop_timer(GPU_start_time, "\t            Total");	
	
	printf("*************** R E S U L T S **************\n");

        printf("GPU Max  :\t%f\n",result[0]);
        printf("CPU Max  :\t%f\n", max_val);

        printf("GPU Min  :\t%f\n",result[1]);
        printf("CPU Min  :\t%f\n", min_val);

        printf("GPU Mean :\t%f\n",result[2]);
        printf("CPU Mean :\t%f\n", mean_val);

        printf("GPU sigma:\t%f\n",result[3]);
	printf("CPU sigma:\t%f\n", stdev_val);

	printf("*******************************************\n");

	// Compute the speedup or slowdown
	if (GPU_time > CPU_time) printf("\nCPU outperformed GPU by %.2fx\n", (float) GPU_time / (float) CPU_time);
	else                     printf("\nGPU outperformed CPU by %.2fx\n", (float) CPU_time / (float) GPU_time);
	
	// Check the correctness of the GPU results
	int num_wrong = 0;
	if (fabs(max_val - result[0]) > 0.000001) num_wrong++;
	if (fabs(min_val - result[1]) > 0.000001) num_wrong++;
	if (fabs(mean_val - result[2]) > 0.000001) num_wrong++;
	if (fabs(stdev_val - result[3]) > 0.000001) num_wrong++;
	
	
	// Report the correctness results
	if (num_wrong) printf("\n%d / %d values incorrect\n", num_wrong, N);
	else           printf("\nAll values correct\n");
        
}


// A GPU kernel that computes the vector sum A + B
// (each thread computes a single value of the result)
__global__ void vector_max(float *max, float *min, float *mean,  int N) {
	// Determine which element this thread is computing
	int block_id = blockIdx.x + gridDim.x * blockIdx.y;
	int thread_id = blockDim.x* block_id + threadIdx.x;
        const int MAX = 0;
        const int MIN = 1;
        const int MEAN = 2;

        //allocate shared memory
        __shared__ float values[threads_per_block][3]; 
        //copy all values to shared memory
        if(thread_id < N){
            values[threadIdx.x][MAX] = max[thread_id];
            values[threadIdx.x][MIN] = min[thread_id];
            values[threadIdx.x][MEAN] = mean[thread_id];
        }else{
            values[threadIdx.x][MAX] = FLT_MIN;
            values[threadIdx.x][MIN] = FLT_MAX;
            values[threadIdx.x][MEAN] = 0;
        }
        __syncthreads();
        
        int th = threads_per_block/2;
        //loop until dead or single element 
        while(th>0){
            
            //for our pair, compute the max
            //if we're a valid value
            if(threadIdx.x < th){
                
                //max calculation
                if(values[threadIdx.x][MAX] < values[threadIdx.x + th][MAX]){
                    //copy
                    values[threadIdx.x][MAX] = values[threadIdx.x + th][MAX];
                }                
                
                //min calculation
                if(values[threadIdx.x][MIN] > values[threadIdx.x + th][MIN]){
                    //copy
                    values[threadIdx.x][MIN] = values[threadIdx.x + th][MIN];
                }                

                //sum calculation
                values[threadIdx.x][MEAN] += values[threadIdx.x + th][MEAN];
                
            }

            th /= 2;
                
            __syncthreads();
        }

        //reduce the values
        if(threadIdx.x == 0 ) {
            max[block_id] = values[0][MAX];
            min[block_id] = values[0][MIN];
            mean[block_id] = values[0][MEAN];
        }
        //now each block has its own min element
        //we want to compute the min element of all blocks
}

// A GPU kernel that computes the standard deviation given a vector, its length, and its mean
// (each thread computes a single value of the result)
__global__ void vector_std(float *vals, int mean, int N, int first) {
	// Determine which element this thread is computing
	int block_id = blockIdx.x + gridDim.x * blockIdx.y;
	int thread_id = blockDim.x* block_id + threadIdx.x;

        //allocate shared memory
        __shared__ float values[threads_per_block];

        //copy all values to shared memory
        if(thread_id < N){
            if(first == 1){
                //if we're the first kernel call, we want to calculate the difference of squares
                values[threadIdx.x] = (vals[thread_id] - mean) * (vals[thread_id] - mean);
            }else{
                //otherwise we just want to sum the original values
                values[threadIdx.x] = vals[thread_id];
            }
        }else{
            values[threadIdx.x] = 0;
        }
        __syncthreads();
        
        int th = threads_per_block/2;
        //loop until dead or single element 
        while(th>0){
            
            //for our pair, compute the max
            //if we're a valid value
            if(threadIdx.x < th){
                   
                //sum of squares of differneces calculation
                values[threadIdx.x]  += values[threadIdx.x + th];

            }

            th /= 2;
                
            __syncthreads();
        }

        //reduce the values
        if(threadIdx.x == 0 ) {
            vals[block_id] = values[0];
        }
        //now each block has its own min element
        //we want to compute the min element of all blocks
}


// Returns the max value of vector A (computed on the GPU)
float *GPU_vector_max(float *A_CPU,  int N) {
	
	long long memory_start_time = start_timer();

	// Allocate GPU memory for the inputs and the result
	int vector_size = N * sizeof(float);
        int vector_length = N;

        float *max_GPU;
        float *min_GPU;
        float *mean_GPU;
        float *std_GPU; 

        //allocate working arrays on GPU
	if (cudaMalloc((void **) &max_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &min_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &mean_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &std_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");

	// Transfer the input vectors to GPU memory
	cudaMemcpy(max_GPU, A_CPU, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(min_GPU, A_CPU, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(mean_GPU, A_CPU, vector_size, cudaMemcpyHostToDevice);
        cudaMemcpy(std_GPU, A_CPU, vector_size, cudaMemcpyHostToDevice);
	
	stop_timer(memory_start_time, "\nGPU:\t  Transfer to GPU");
	
	// Determine the number of thread blocks
	int num_blocks = (int) ((float) (N + threads_per_block - 2) / (float) threads_per_block);
	int max_blocks_per_dimension = 65535;
        dim3 grid_size(1,1,1);

        if(num_blocks > max_blocks_per_dimension){
            grid_size.x = max_blocks_per_dimension; grid_size.y = num_blocks/max_blocks_per_dimension + 1;                   
	}else{
            grid_size.x = num_blocks;                   
        }

        float *result = (float *)malloc(4 * sizeof(float));
        
	// Execute the kernel to compute the vector max on the GPU
	long long first_kernel_start_time = start_timer();
      
       
        //while N > threads per block
        while(num_blocks > 0){
 
            vector_max <<< grid_size , threads_per_block >>> (max_GPU, min_GPU, mean_GPU, N);
            cudaThreadSynchronize();  // this is only needed for timing purposes
            
            //resize N
            N = num_blocks;

            // Determine the new number of thread blocks
            num_blocks = (int) ((float) (N + threads_per_block - 2) / (float) threads_per_block);
            
            if(num_blocks > max_blocks_per_dimension){
                grid_size.x = max_blocks_per_dimension; grid_size.y =  num_blocks/max_blocks_per_dimension + 1;               
            }else{
                grid_size.x = num_blocks;            
            }                   
            
	}
        stop_timer(first_kernel_start_time, "\t Min,Max,Mean kernel execution");

        // Transfer the result from the GPU to the CPU
	memory_start_time = start_timer();
 	cudaMemcpy(&result[0], max_GPU, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result[1], min_GPU, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result[2], mean_GPU, sizeof(float), cudaMemcpyDeviceToHost);

	//calculate mean
        result[2] = result[2]/vector_length;
        
        stop_timer(memory_start_time, "\t Transfer from GPU");
	
	// Free the GPU memory
	cudaFree(max_GPU);
        cudaFree(min_GPU);
        cudaFree(mean_GPU);


        //second launch to calculate std deviation in two passes

        //reset N
        N = vector_length;

	// Determine the number of thread blocks
	num_blocks = (int) ((float) (N + threads_per_block - 2) / (float) threads_per_block);
        
        if(num_blocks > max_blocks_per_dimension){
            grid_size.x = max_blocks_per_dimension; grid_size.y = num_blocks/max_blocks_per_dimension + 1;                   
	}else{
            grid_size.x = num_blocks; 
            grid_size.y = 1;
        }
        int first = 1;

        long long second_kernel_start_time = start_timer();

        while(num_blocks > 0){

            vector_std <<< grid_size , threads_per_block >>> (std_GPU, result[2], N, first);
            cudaThreadSynchronize();  // this is only needed for timing purposes

            //make sure that we do not do mean subtraction and squaring again.
            first++;

            //resize N
            N = num_blocks;

            // Determine the new number of thread blocks
            num_blocks = (int) ((float) (N + threads_per_block - 2) / (float) threads_per_block);
            
            if(num_blocks > max_blocks_per_dimension){
                grid_size.x = max_blocks_per_dimension; grid_size.y =  num_blocks/max_blocks_per_dimension + 1;               
            }else{
                grid_size.x = num_blocks;            
            }                   
            
	}
        stop_timer(second_kernel_start_time, "\tStdev Kernel execution");
        

	// Check for kernel errors
	cudaError_t error = cudaGetLastError();
	if (error) {
		char message[256];
		sprintf(message, "CUDA error: %s", cudaGetErrorString(error));
		die(message);
	}
	
	memory_start_time = start_timer();
        
        //copy result from GPU
	cudaMemcpy(&result[3], std_GPU, sizeof(float), cudaMemcpyDeviceToHost);

        //do std deviation calc
        result[3] = sqrt(result[3]/vector_length);

        stop_timer(memory_start_time, "\tTransfer from GPU");
        
        //free std_GPU
        cudaFree(std_GPU);

        return result;
}


// Returns the vector sum A + B
float *CPU_add_vectors(float *A, float *B, int N) {	
	// Allocate memory for the result
	float *C = (float *) malloc(N * sizeof(float));
	if (C == NULL) die("Error allocating CPU memory");

	// Compute the sum;
	for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
	
	// Return the result
	return C;
}


// Returns a randomized vector containing N elements
float *get_random_vector(int N) {
	if (N < 1) die("Number of elements must be greater than zero");
	
	// Allocate memory for the vector
	float *V = (float *) malloc(N * sizeof(float));
	if (V == NULL) die("Error allocating CPU memory");
	
	// Populate the vector with random numbers
	for (int i = 0; i < N; i++) {
            V[i] = (float) rand() / (float) rand();
        }
	// Return the randomized vector
	return V;
}


// Returns the current time in microseconds
long long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, char *name) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
	printf("%s: %.5f sec\n", name, ((float) (end_time - start_time)) / (1000 * 1000));
	return end_time - start_time;
}

float cpu_max(float *A, int N){ 
        float max = A[0];
        for(int i = 1; i<N; i++){
            if(A[i] > max){
                max = A[i];
            }
        }

	return  max; 
}

float cpu_min(float *A, int N){
        float min = A[0];
        for(int i = 1; i<N; i++){
            if(A[i] < min){
                min = A[i];
            }
        }

	return  min; 
}

float cpu_mean(float *A, int N){ 
    float mean = A[0];
        for(int i = 1; i<N; i++){
                mean += A[i];
	}
        return  mean/N; 
}

float cpu_stdev(float *A, int N){ 

	//get average
	float mean = cpu_mean(A,N);

	//get sum of squares of differences
	float sum_diff_squared = 0.0;
	for(int i=0; i<N; ++i){
		sum_diff_squared += (A[i]-mean) * (A[i]-mean);
	} //end loop

	//return square root of average
	return sqrt(sum_diff_squared / N);
}//eo cpu_stdev


// Prints the specified message and quits
void die(char *message) {
	printf("%s\n", message);
	exit(1);
}
