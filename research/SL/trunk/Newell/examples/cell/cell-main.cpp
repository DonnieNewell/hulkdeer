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
#include "Cluster.h"
#include "Decomposition.h"
#include "Balancer.h"
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

enum MPITagType {xDim = 0, xLength = 1, xChildren = 2, xDevice = 3, xData = 4, xNumBlocks = 5, xOffset = 6};
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

void sendDataToNode(int rank, int device, SubDomain3D& s){

  
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
	MPI_Isend((void*)&device, 1, MPI_INT, rank,xDevice, MPI_COMM_WORLD, &reqs[0]);
	#ifdef NOT_DEFINED
		printf("[%d] sending %dD data to Node %d.\n",0,numDim,rank);
	#endif
	MPI_Isend((void*)&numDim, 1, MPI_INT, rank,xDim, MPI_COMM_WORLD, &reqs[1]);
        //second send size of each dimension
	MPI_Isend((void*)length, 3, MPI_INT, rank,xLength, MPI_COMM_WORLD, &reqs[2]);
	MPI_Isend((void*)offset, 3, MPI_INT, rank,xOffset, MPI_COMM_WORLD, &reqs[3]);
	#ifdef NOT_DEFINED
		printf("[%d] sending [%d][%d][%d] data to Node %d.\n",0,length[2],length[1],length[0],rank);
		printf("[%d] sending %dX%dX%d offsets to Node %d.\n",0,offset[2],offset[1],offset[0],rank);
	#endif

	//third send data  
//	//first we have to stage the data into contiguous memory
	int total_size=1;
	for(int i=0; i<numDim; ++i){
		total_size *= length[i];
	}
//	int *staged_data = new int[total_size];
//	for(int i=0; i<total_size; ++i){
//		staged_data[i] = space3D[offset[2]+i/(K*J)][offset[1]+(i%(J*K))/J][offset[0]+(i%J)];
//	}//end for
	MPI_Isend((void*)s.getBuffer(), total_size, MPI_INT, rank,xData, MPI_COMM_WORLD, &reqs[4]);
	
	
	//wait for everything to finish
	MPI_Waitall(5,reqs,MPI_STATUSES_IGNORE);
	//clean up memory
	//delete staged_data;
}

void getNumberOfChildren(int& numChildren){
	#ifdef NOT_DEFINED
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
	#ifdef NOT_DEFINED
		printf("sending number of children:%d.\n",numChildren);
	#endif
	MPI_Request req;
	MPI_Isend((void*)&numChildren, 1, MPI_INT, dest_rank, xChildren, MPI_COMM_WORLD, &req);
	
	
	//wait for everything to finish
	MPI_Waitall(1,&req,MPI_STATUSES_IGNORE);
	
}

void receiveNumberOfChildren(int numTasks, Cluster &cluster){
	#ifdef NOT_DEFINED
		printf("receiving number of children.\n");
	#endif
	MPI_Request *reqs = (MPI_Request*)malloc(sizeof(MPI_Request)*(numTasks-1));	
	

	int* numChildren = (int*)malloc((numTasks-1)*sizeof(int));

	for(int i=0; i<numTasks-1; i++){
		//receive next count
		MPI_Irecv((void*)&(numChildren[i]), 1, MPI_INT, i+1, xChildren, MPI_COMM_WORLD, &(reqs[i]));
	}
	MPI_Waitall(numTasks-1,reqs,MPI_STATUSES_IGNORE);
	for(int task=0; task < numTasks-1; task++){
#ifdef DEBUG
		printf("[0]: child [%d] has %d children.\n",task+1, numChildren[task]);
#endif
		cluster.getNode(task+1).setNumChildren(numChildren[task]);
	}

	free(reqs);
	reqs=NULL;
	free(numChildren);
	numChildren = NULL;
}

void sendData(Node& n){
	//count how many task blocks, total, are going to be sent
	int total = n.numSubDomains();
	for(int child=0; child < n.getNumChildren(); ++child){
		total += n.getChild(child).numSubDomains();
	}

	//send node number of blocks
	MPI_Request req;
	MPI_Isend((void*)&total, 1, MPI_INT, n.getRank(),xNumBlocks,MPI_COMM_WORLD, &req);

	int device = -1;
	for(int i=0; i< n.numSubDomains(); ++i){
		sendDataToNode(n.getRank(),device, n.getSubDomain(i));
	}
	for(int child=0; child < n.getNumChildren(); ++child){
		for(int i=0; i< n.getChild(child).numSubDomains(); ++i){
			sendDataToNode(n.getRank(), device, n.getChild(child).getSubDomain(i));
		}
		
	}
	//wait for first send to finish
	MPI_Waitall(1,&req, MPI_STATUSES_IGNORE);
}


/* output variables: buf, size */
void receiveDataFromNode(int rank,int& device, SubDomain3D &s){
	MPI_Request reqs[5];
	int numDim = 0;
	int length[3];
	int offset[3];
        //int *buffer = NULL;
	//receive dimensionality of data
	MPI_Irecv((void*)&device, 1, MPI_INT, rank, xDevice, MPI_COMM_WORLD,  &reqs[0]);
	#ifdef NOT_DEFINED
		fprintf(stderr,"[%d] receiving dimensionality from Node %d.\n",rank,0);
	#endif
	MPI_Irecv((void*)&numDim, 1, MPI_INT, rank, xDim, MPI_COMM_WORLD,  &reqs[1]);

	//receive size of data
	#ifdef NOT_DEFINED 
		fprintf(stderr,"[%d] receiving size of data from Node %d.\n",rank,0);
	#endif
	MPI_Irecv((void*)length, 3, MPI_INT, rank, xLength, MPI_COMM_WORLD,  &reqs[2]);
	MPI_Irecv((void*)offset, 3, MPI_INT, rank, xOffset, MPI_COMM_WORLD,  &reqs[3]);

	MPI_Waitall(4,reqs,MPI_STATUSES_IGNORE);

	int size=1;
	for(int i =0; i<numDim; ++i){
		s.setLength(i, length[i]);
		s.setOffset(i, offset[i]);
		size *= length[i];
	}

	//allocates data memory and sets up 2d and 3d data pointers
	//initData(length);

	//needs to be set by compiler. DTYPE maybe?
	int* buf = new int[size];
	#ifdef NOT_DEFINED
		fprintf(stderr,"[%d] about to receive data from Node %d.\n",rank,0);
	#endif
	//MPI_INT needs to be set by compiler. DTYPE maybe?
	MPI_Irecv((void*)buf, size, MPI_INT, rank, xData, MPI_COMM_WORLD,  &reqs[4]);

	//wait for everything to finish
	MPI_Waitall(1,reqs,MPI_STATUSES_IGNORE);
	#ifdef NOT_DEFINED 
		printf("received %dD data from %d.\n",numDim,rank); 
		printf("received [%d][%d][%d] length data from  %d.\n",length[2],length[1],length[0],rank); 
	#endif

	s.setBuffer(buf);

}

void receiveData(int rank, Node& n){
	//receive number of task blocks that will be sent
	int numTaskBlocks=0;
	MPI_Status stat;
	MPI_Recv((void*)&numTaskBlocks,1, MPI_INT, rank, xNumBlocks, MPI_COMM_WORLD, &stat);
	
	for(int block = 0; block<numTaskBlocks; ++block){
		SubDomain3D s;
		int device = -1;
		receiveDataFromNode(rank, device, s);
		if(-1 == device)
		{
			//add block to cpu queue
			n.addSubDomain(s);
		}
		else
		{
			//add block to gpu queue
			n.getChild(device).addSubDomain(s);
		}
	}
}

void processSubDomain(SubDomain3D &task, int timesteps, int bornMin, int bornMax, int dieMin, int dieMax){
	//DTYPE?
	int* buff = task.getBuffer();
	int depth = task.getLength(0);
	int height = task.getLength(1);
	int width = task.getLength(2);
	runCell(buff, depth, height, width, timesteps, bornMin, bornMax, dieMin, dieMax);

}

int bornMin = 5, bornMax = 8;
int dieMax = 3, dieMin = 10;

int main(int argc, char** argv)
{
	int numTasks, rank, rc, *buffer, buffSize,deviceCount=0;
	Node myWork;

	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS){
		fprintf(stderr, "Error initializing MPI.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	myWork.setRank(rank);
	
	getNumberOfChildren(deviceCount);
	printf("[%d] children:%d\n",rank,deviceCount);

	if(0==rank){
	
		//get the number of children from other nodes
		Cluster cluster(numTasks);
		cluster.getNode(0).setNumChildren(deviceCount);
		receiveNumberOfChildren(numTasks, cluster);
	
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
		decomp.decompose(data,3,numElements);
	
		/* now perform the load balancing, assigning task blocks to each node */
		Balancer lb;
		lb.balance(cluster,decomp);
		
		printCluster(cluster);
		#ifdef DEBUG
  			fprintf(stderr,"[%d] decomposed data into %d chunks.\n",rank, decomp.getNumSubDomains());
 			fprintf(stderr,"[%d] sending data.",rank);
		#endif
		//this is commented out so that it will compile and we can test that the domain 
		//decomposition is working correctly
		//then we will handle the load balancing aspect
		//once we have the machine graph represented, then we can send the data.
		for(int node=1; node < cluster.getNumNodes(); ++node){
			sendData(cluster.getNode(node));
		}
	}
	else{
		//send number of children to root
		sendNumberOfChildren(0,deviceCount);

		timesteps = atoi(argv[4]);
		pyramid_height= atoi(argv[5]);
		receiveData(0, myWork);
#ifdef DEBUG
		fprintf(stderr, "[%d] processing %d task blocks.\n",rank,myWork.numSubDomains());
#endif
		for(int task=0; task<myWork.numSubDomains(); ++task){
			processSubDomain(myWork.getSubDomain(task),timesteps, bornMin, bornMax, dieMin, dieMax);
		}
		for(int child=0; child<myWork.getNumChildren(); ++child){
#ifdef DEBUG
			fprintf(stderr, "[%d] child [%d] processing %d task blocks.\n",rank,child,myWork.getChild(child).numSubDomains());
#endif
		}
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

