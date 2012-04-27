/* -*- Mode: C ; indent-tabs-mode: nil ; c-file-style: "stroustrup" -*-

    CS 6620 - Compilers
    Stencil App Language Project
    Authors: Greg Faust, Sal Valente, derived from code by Jiayuan Meng
    

    File:   pathfinder-main.cpp     Contains a main routine to drive the pathfinder example.
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cell.h"
#include "Decomposition.h"
#include "../Model.h"
#include "mpi.h"

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

int J, K, L;
int* data;
int** space2D;
int*** space3D;
#define M_SEED 9
int pyramid_height;
int timesteps;

const int dim_tag = 0;
const int length_tag = 1;
const int children_tag = 2;
const int data_tag = 3;
// #define BENCH_PRINT

void
initData(int length[3] )
{
	#ifdef DEBUG
		fprintf(stderr, "initializing data only.\n");
	#endif
	J = length[0];
	K = length[1];
	L = length[2];
	#ifdef DEBUG
		fprintf(stderr, "allocating data[%d][%d][%d].\n",L,K,J);
	#endif
	data = new int[J*K*L];
	#ifdef DEBUG
		fprintf(stderr, "allocating space2D.\n");
	#endif
        space2D = new int*[J*K];
	#ifdef DEBUG
		fprintf(stderr, "allocating space3D.\n");
	#endif
	space3D = new int**[J];
	#ifdef DEBUG
		fprintf(stderr, "initializing space2D only.\n");
	#endif
	for(int n=0; n<J*K; n++)
          space2D[n]=data+L*n;
	#ifdef DEBUG
		fprintf(stderr, "initializing space3D only.\n");
	#endif
	for(int n=0; n<J; n++)
          space3D[n]=space2D+K*n;
}

void
init(int argc, char** argv)
{
	#ifdef DEBUG
		fprintf(stderr, "initializing data for root node.\n");
	#endif
	if(argc==6){
		J = atoi(argv[1]);
		K = atoi(argv[2]);
                L = atoi(argv[3]);
                timesteps = atoi(argv[4]);
                pyramid_height=atoi(argv[5]);
	}else{
                printf("Usage: cell dim3 dim2 dim1 timesteps pyramid_height\n");
                exit(0);
        }
	data = new int[J*K*L];
        space2D = new int*[J*K];
	space3D = new int**[J];
	for(int n=0; n<J*K; n++)
          space2D[n]=data+L*n;
	for(int n=0; n<J; n++)
          space3D[n]=space2D+K*n;

	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < J*K*L; i++)
            data[i] = rand()%2;

}



void printResults(int* data, int J, int K, int L)
{
    int total = J*K*L;
    for(int n = 0; n < total; n++){
        printf("%d ", data[n]);
    }
    printf("\n");
}

void sendDataToNode(int rank, SubDomain3D s){
  
	//first send number of dimensions
	int numDim = 0;
	MPI_Request reqs[5];
	int length[3];
	int offset[3];
	for(int i=0;i<3;++i){
		/* DEBUG: just send first work block */
		length[i]=s.getLength(i);
		offset[i]=s.getOffset(i);
		if(length[i]>0) numDim++;
	}  
	#ifdef DEBUG
		printf("[%d] sending %dD data to Node %d.\n",0,numDim,rank);
	#endif
	MPI_Isend((void*)&numDim, 1, MPI_INT, rank,dim_tag, MPI_COMM_WORLD, &reqs[0]);
        //second send size of each dimension
	MPI_Isend((void*)length, 3, MPI_INT, rank,length_tag, MPI_COMM_WORLD, &reqs[1]);
	#ifdef DEBUG
		printf("[%d] sending [%d][%d][%d] data to Node %d.\n",0,length[2],length[1],length[0],rank);
		printf("[%d] sending %dX%dX%d offsets to Node %d.\n",0,offset[2],offset[1],offset[0],rank);
	#endif

	//third send data  
	//first we have to stage the data into contiguous memory
	int total_size=1;
	for(int i=0; i<numDim; ++i){
		total_size *= length[i];
	}
	//no need to stage the data now that the subdomain class holds a pointer to the data
	//int *staged_data = new int[total_size];
	//for(int i=0; i<total_size; ++i){
	//	staged_data[i] = space3D[offset[2]+i/(K*J)][offset[1]+(i%(J*K))/J][offset[0]+(i%J)];
	//}//end for
	MPI_Isend(s.getBuffer(), total_size, MPI_INT, rank,data_tag, MPI_COMM_WORLD, &reqs[2]);
	
	
	//wait for everything to finish
	MPI_Waitall(3,reqs,MPI_STATUSES_IGNORE);
	//clean up memory
	//delete staged_data;
}

void getNumberOfChildren(int& numChildren){
	#ifdef DEBUG
		printf("getting the number of children...\n");
	#endif
	/* check to see how many NVIDIA GPU'S ARE AVAILABLE */
	cudaError_t err = cudaGetDeviceCount(&numChildren);
	if(cudaSuccess != err){
		fprintf(stderr, "error detecting cuda-enabled devices\n");
		exit(1);
	}
	
}
void sendNumberOfChildren(const int dest_rank, const int numChildren){
	#ifdef DEBUG
		printf("sending number of children:%d.\n",numChildren);
	#endif
	MPI_Request req;
	MPI_Isend((void*)&numChildren, 1, MPI_INT, dest_rank, children_tag, MPI_COMM_WORLD, &req);
	
	
	//wait for everything to finish
	MPI_Waitall(1,&req,MPI_STATUSES_IGNORE);
	
}

void receiveNumberOfChildren(int numTasks, int *numChildren){
	#ifdef DEBUG
		printf("receiving number of children.\n");
	#endif
	MPI_Request *reqs = (MPI_Request*)malloc(sizeof(MPI_Request)*(numTasks-1));	
	
	if(numChildren != NULL) free(numChildren);

	numChildren = (int*)malloc((numTasks-1)*sizeof(int));

	for(int i=0; i<numTasks-1; i++){
		//receive next count
		MPI_Irecv((void*)&(numChildren[i]), 1, MPI_INT, i+1, children_tag, MPI_COMM_WORLD, &(reqs[i]));
	}
	MPI_Waitall(numTasks-1,reqs,MPI_STATUSES_IGNORE);
#ifdef DEBUG
	for(int task=0; task < numTasks-1; task++){
		printf("[0]: child [%d] has %d children.\n",task+1, numChildren[task]);
	}
#endif
	free(reqs);
	reqs=NULL;
}

void sendData(int rank, Node& n){
	for(int i=0; i< n.numSubDomains(); ++i){
		sendDataToNode(r, n.getSubDomain(i));
	}
}

/* output variables: buf, size */
void receiveData(int rank, int* buf, int *size){
	MPI_Request reqs[5];
	int numDim = 0;
	int length[3];
	//receive dimensionality of data
	#ifdef DEBUG
		fprintf(stderr,"[%d] receiving dimensionality from Node %d.\n",rank,0);
	#endif
	MPI_Irecv((void*)&numDim, 1, MPI_INT, 0, dim_tag, MPI_COMM_WORLD,  &reqs[0]);

	//receive size of data
	#ifdef DEBUG
		fprintf(stderr,"[%d] receiving size of data from Node %d.\n",rank,0);
	#endif
	MPI_Irecv((void*)length, 3, MPI_INT, 0, length_tag, MPI_COMM_WORLD,  &reqs[1]);

	*size=1;
	for(int i =0; i<numDim; ++i){
		(*size) *= length[i];
	}

	MPI_Waitall(2,reqs,MPI_STATUSES_IGNORE);
	//allocates data memory and sets up 2d and 3d data pointers
	initData(length);

	#ifdef DEBUG
		fprintf(stderr,"[%d] about to receive data from Node %d.\n",rank,0);
	#endif
	MPI_Irecv((void*)buf, *size, MPI_INT, 0, data_tag, MPI_COMM_WORLD,  &reqs[2]);

	//wait for everything to finish
	MPI_Waitall(1,reqs,MPI_STATUSES_IGNORE);
	#ifdef DEBUG
		printf("[%d] received %dD data from %d.\n",rank,numDim,0); 
		printf("[%d] received [%d][%d][%d] length data from  %d.\n",rank,length[2],length[1],length[0],0); 
	#endif

}


int bornMin = 5, bornMax = 8;
int dieMax = 3, dieMin = 10;

int main(int argc, char** argv)
{
	int numTasks, rank, rc, *buffer, buffSize,deviceCount=0;
	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS){
		fprintf(stderr, "Error initializing MPI.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	getNumberOfChildren(deviceCount);
	printf("children:%d\n",deviceCount);

	if(0==rank){
	
		//get the number of children from other nodes
		int *numChildren=NULL;
		receiveNumberOfChildren(numTasks, numChildren);
	
 		#ifdef  DEBUG
			fprintf(stderr, "[%d] initializing data.\n",rank);
		#endif

  		init(argc, argv); //initialize data

		#ifdef DEBUG
  			fprintf(stderr,"[%d] decomposing data.\n",rank);
		#endif

		/* perform domain decomposition */
		Decomposition decomp;
		int numElements[3] = {J,K,L};
		decomp.decompose(3,numElements);

		#ifdef DEBUG
  			fprintf(stderr,"[%d] decomposed data into %d chunks.\n",rank, decomp.getNumSubDomains());
 			fprintf(stderr,"[%d] sending data.",rank);
		#endif
		sendData(numTasks-1,decomp, data, J*K*L);
	}
	else{
		//send number of children to root
		sendNumberOfChildren(0,deviceCount);

		timesteps = atoi(argv[4]);
		pyramid_height= atoi(argv[5]);
		receiveData(rank, buffer,&buffSize);
	}
#ifdef STATISTICS
	for (int i=40; i<=J; i += 20)
	{
		// Set iteration count so that kernel is called at least 30 times.
		// The maximum pyramid height is 3, so iterations = 90.
		//runCell(data, i, i, i, 90, bornMin, bornMax, dieMin, dieMax);
	}
#else
	runCell(data, J, K, L, timesteps, bornMin, bornMax, dieMin, dieMax);

#ifdef BENCH_PRINT
	printResults(data, J, K, L);
#endif
#endif

	MPI_Finalize();
	delete [] data;
	delete [] space2D;
	delete [] space3D;

	return 0;
}

