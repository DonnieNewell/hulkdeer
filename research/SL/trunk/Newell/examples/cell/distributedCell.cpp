#include <stdlib.h>
#include <stdio.h>
#include <limits>
#include "cell.h"
#include "ompCell.h"
#include "Cluster.h"
#include "Decomposition.h"
#include "Balancer.h"
#include "../Model.h"
#include "mpi.h"
#ifndef WIN32
#include <sys/time.h>
#else
#include<time.h>
#endif

#define DTYPE int
#define PYRAMID_HEIGHT 1

enum MPITagType {
  xDim        = 0, xLength  = 1, xChildren    = 2, 
  xDevice     = 3, xData    = 4, xNumBlocks   = 5, 
  xOffset     = 6, xWeight  = 7, xWeightIndex = 8,
  xEdgeWeight = 9, xId      = 10  };

void sendDataToNode(int rank, int device, SubDomain3D* s)
{
  //first send number of dimensions
  int numDim  = 0 ;
  MPI_Request reqs[6] ;
  int length[3] ;
  int offset[3] ;
  const int* tmpId = s->getId();
  MPI_Isend(  (void*)tmpId , 3, MPI_INT, rank, xId,      MPI_COMM_WORLD, &reqs[5]);
  MPI_Isend(  (void*)&device, 1, MPI_INT, rank, xDevice,  MPI_COMM_WORLD, &reqs[0]);
  for(int i=0; i < 3; ++i)
  {
    length[i] = s->getLength(i) ;
    offset[i] = s->getOffset(i) ;
    
    if(length[i] > 0) numDim++;
  }  
  MPI_Isend(  (void*)&numDim,       1, MPI_INT, rank, xDim,     MPI_COMM_WORLD, &reqs[1]);
  MPI_Isend(  (void*)length,        3, MPI_INT, rank, xLength,  MPI_COMM_WORLD, &reqs[2]);
  MPI_Isend(  (void*)offset,        3, MPI_INT, rank, xOffset,  MPI_COMM_WORLD, &reqs[3]);

  //third send data  
  //first we have to stage the data into contiguous memory
  int total_size = 1;
  for(int i=0; i < numDim; ++i)
  {
    total_size *= length[i];
  }
  MPI_Isend(  (void*)s->getBuffer(), total_size, MPI_INT, rank,xData, MPI_COMM_WORLD, &reqs[4]);

  MPI_Waitall(6, reqs, MPI_STATUSES_IGNORE);
}

void getNumberOfChildren(int& numChildren){
#ifdef NOT_DEFINED
  printf("getting the number of children...\n");
#endif
  /* check to see how many NVIDIA GPU'S ARE AVAILABLE */
  cudaError_t err = cudaGetDeviceCount(&numChildren);
  if(cudaSuccess == cudaErrorNoDevice)
  {
    numChildren = 0;
  }
  else if(cudaSuccess != err){
    fprintf(stderr, "error detecting cuda-enabled devices\n");
    numChildren = 0;
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
      sendDataToNode(n.getRank(), child, n.getChild(child).getSubDomain(i));
    }

  }
  //wait for first send to finish
  MPI_Waitall(1,&req, MPI_STATUSES_IGNORE);
}

void benchmarkNode(Node& n, SubDomain3D* s)
{
  struct timeval start, end;
  double total_sec = 0.0;	

  gettimeofday(&start, NULL);                                       
  //send task block to every device on that node
  sendDataToNode(n.getRank(), -1, s);
  gettimeofday(&end, NULL);                                       

  total_sec = ((end.tv_sec - start.tv_sec) +             
      (end.tv_usec - start.tv_usec)/1000000.0);                       

  fprintf(stderr, "it takes %f sec to send one task to node: %d.\n",total_sec,n.getRank());
  //how fast is the connection between root and child nodes
  //multiply by 2 to account for there and back
  n.setEdgeWeight(1/(2*total_sec));
#ifdef DEBUG
  fprintf(stderr, "benchmarkNode(node.rank:%d, edgeWeight:%f blocks/sec): sent data to the node.\n",n.getRank(), n.getEdgeWeight());
#endif
  //receive results for each device
  int total = n.getNumChildren()+1;

  MPI_Request req[2];
  double *task_per_sec= new double[total];
  double *edge_weight= new double[total-1];

  MPI_Irecv((void*)task_per_sec, total, MPI_DOUBLE, n.getRank(), xWeight, MPI_COMM_WORLD, &req[0]);
  MPI_Irecv((void*)edge_weight, total-1, MPI_DOUBLE, n.getRank(), xEdgeWeight, MPI_COMM_WORLD, &req[1]);

  MPI_Waitall(2,req,MPI_STATUSES_IGNORE);

  //set the appropriate fields in the node and its children
  for(int device = 0; device<total; ++device)
  {
    double weight = task_per_sec[device];
    if(device == 0)
    {
      //the first weight is for the cpu
      fprintf(stderr,"setting node[%d] weight to %f.\n",n.getRank(),weight);
      n.setWeight(weight);
    }
    else
    {
      double edgeWeight = edge_weight[device-1];
      fprintf(stderr,"setting node[%d].child[%d] weight to %f.\n",n.getRank(),device-1,weight);
      n.getChild(device-1).setWeight(weight);
      n.getChild(device-1).setEdgeWeight(edgeWeight);
    }
  }

  //clean up
  delete [] task_per_sec;
  task_per_sec = NULL;
  delete [] edge_weight;
  edge_weight = NULL;
}

/* output variables: buf, size */
SubDomain3D* receiveDataFromNode(int rank,int& device){
  MPI_Request reqs[5];
  int numDim =  0 ;
  int id[3]     = {-1,-1,-1} ;
  int length[3]   ;
  int offset[3]   ;
  //int *buffer = NULL;
  //receive dimensionality of data
  MPI_Irecv((void*)id, 3, MPI_INT, rank, xId, MPI_COMM_WORLD,  &reqs[4]);
  MPI_Irecv((void*)&device, 1, MPI_INT, rank, xDevice, MPI_COMM_WORLD,  &reqs[0]);
  MPI_Irecv((void*)&numDim, 1, MPI_INT, rank, xDim, MPI_COMM_WORLD,  &reqs[1]);

  //receive size of data
  MPI_Irecv((void*)length, 3, MPI_INT, rank, xLength, MPI_COMM_WORLD,  &reqs[2]);
  MPI_Irecv((void*)offset, 3, MPI_INT, rank, xOffset, MPI_COMM_WORLD,  &reqs[3]);

  MPI_Waitall(5,reqs,MPI_STATUSES_IGNORE);

  SubDomain3D *s = new SubDomain3D(id, offset[0], length[0], offset[1], length[1], offset[2], length[2]);
  int size=1;
  for(int i =0; i < numDim; ++i)
  {
    s->setLength(i, length[i]);
    s->setOffset(i, offset[i]);
    size *= length[i];
  }

  //allocates data memory and sets up 2d and 3d data pointers
  //initData(length);

  //needs to be set by compiler. DTYPE maybe?
  //we assume that if the buffer is already allocated, that the size is correct.
  //if(s.getBuffer()==NULL)
  //		s.setBuffer(new int[size]);

  //MPI_INT needs to be set by compiler. DTYPE maybe?
  MPI_Irecv((void*)s->getBuffer(), size, MPI_INT, rank, xData, MPI_COMM_WORLD,  &reqs[0]);

  //wait for everything to finish
  MPI_Waitall(1,reqs,MPI_STATUSES_IGNORE);
  return s;
}
void processSubDomain(int rank, int device, SubDomain3D *task, int timesteps, int bornMin, int bornMax, int dieMin, int dieMax){
  //DTYPE?
  DTYPE* buff = task->getBuffer();
  int depth = task->getLength(0);
  int height = task->getLength(1);
  int width = task->getLength(2);
  struct timeval start, end;
  if(-1==device)
  {
    //run on CPU
    runOMPCell(buff, depth, height, width, timesteps, bornMin, bornMax, dieMin, dieMax, device);
    //usleep(80000);

  }
  else
  {
    //run on GPU
    gettimeofday(&start,NULL);
    runCell(buff, depth, height, width, timesteps, bornMin, bornMax, dieMin, dieMax,device);
    gettimeofday(&end,NULL);
    double kerneltime = ((end.tv_sec - start.tv_sec) +             
        (end.tv_usec - start.tv_usec)/1000000.0);                       
  }
}

void receiveData(int rank, Node& n, bool processNow, int iterations=0, int bornMin=0, int bornMax=0, int dieMin=0, int dieMax=0){
  //receive number of task blocks that will be sent
  int numTaskBlocks=0;
  MPI_Status stat;
  MPI_Recv((void*)&numTaskBlocks,1, MPI_INT, rank, xNumBlocks, MPI_COMM_WORLD, &stat);
  struct timeval start, end;
  double receiveDataTime=0.0, processBlockTime=0.0;
  for(int block = 0; block<numTaskBlocks; ++block){
    SubDomain3D* s=NULL;
    int device = -1;
    gettimeofday(&start, NULL);
    s= receiveDataFromNode(rank, device);
    gettimeofday(&end, NULL);
    receiveDataTime += ((end.tv_sec - start.tv_sec) +             
        (end.tv_usec - start.tv_usec)/1000000.0);                       
    if(-1 == device)
    {
      if(processNow)
      {
        gettimeofday(&start, NULL);
        processSubDomain(rank,-1, s,iterations, bornMin, bornMax, dieMin, dieMax);
        gettimeofday(&end, NULL);
        processBlockTime+= ((end.tv_sec - start.tv_sec) +             
            (end.tv_usec - start.tv_usec)/1000000.0);                       
      }
      //add block to cpu queue
      n.addSubDomain(s);
    }
    else
    {
      if(processNow)
      {
        gettimeofday(&start, NULL);
        processSubDomain(rank, device, s, iterations, bornMin, bornMax, dieMin, dieMax);
        gettimeofday(&end, NULL);
        processBlockTime+= ((end.tv_sec - start.tv_sec) +             
            (end.tv_usec - start.tv_usec)/1000000.0);                       
      }
      //add block to gpu queue
      n.getChild(device).addSubDomain(s);
    }
  }
  fprintf(stderr,"[%d] comm. time %f, process time %f.\n",n.getRank(),receiveDataTime,processBlockTime);
  runCellCleanup();
}

double benchmarkPCIBus(SubDomain3D* pS, int gpuIndex)
{
  struct timeval start, end;
  double total=0.0;
  gettimeofday(&start, NULL);
  DTYPE* devBuffer  = NULL;
  int    currDevice = -1  ;
  cudaGetDevice(&currDevice);
  if(currDevice != gpuIndex)
  {
    if(cudaSetDevice(gpuIndex)!= cudaSuccess)
    {
      fprintf(stderr,"ERROR: couldn't set device to %d\n",gpuIndex);
      return -1.0;
    }
  }
  size_t size = sizeof(DTYPE)*pS->getLength(0)*pS->getLength(1)*pS->getLength(2);
  cudaMalloc(&devBuffer, size);
  cudaMemcpy((void*)devBuffer,(void*)pS->getBuffer(), size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)pS->getBuffer(),(void*)devBuffer, size, cudaMemcpyDeviceToHost);
  cudaFree(devBuffer);
  devBuffer=NULL;
  gettimeofday(&end, NULL);

  total = ((end.tv_sec - start.tv_sec) +             
      (end.tv_usec - start.tv_usec)/1000000.0);                       

  return 1/total;
}

void benchmarkMyself(Node& n,SubDomain3D* pS, int timesteps, int bornMin, int bornMax, int dieMin, int dieMax)
{

  //printCudaDevices();
  //receive results for each device
  int total = n.getNumChildren()+1;
  MPI_Request req[2];
  double *weight = new double[total];
  double *edgeWeight = new double[total-1];
  SubDomain3D *s=NULL;
  int rank = -2;
  if(pS==NULL)
  {
    s = receiveDataFromNode(0, rank);

    if(-1 != rank)
    {
      //error
      fprintf(stderr, "data for benchmark should be sent to device: -1, not:%d\n",rank);
    }
  }
  else
  {
    s = pS;
  }
  for(int device=0; device < total; ++device)
  {
    int iterations = 100;
    struct timeval start, end;
    double total_sec = 0.0;	

    gettimeofday(&start, NULL);                                       

    for(int itr=0; itr<iterations; ++itr)
    {
      processSubDomain(rank, device-1, s, timesteps, bornMin, bornMax, dieMin, dieMax);
    }

    gettimeofday(&end, NULL);                                       

    total_sec = ((end.tv_sec - start.tv_sec) +             
        (end.tv_usec - start.tv_usec)/1000000.0);                       
    weight[device] =   iterations/total_sec ;
    fprintf(stderr,"[%d]device:%d of %d processes %f iter/sec.\n", n.getRank(),device-1, total, weight[device]);
    if(device==0)
    {
      n.setWeight(weight[device]);
      n.setEdgeWeight(numeric_limits<double>::max());
    }
    else
    {
      n.getChild(device-1).setWeight(weight[device]);
      edgeWeight[device-1] =   benchmarkPCIBus(s,device-1);
      n.getChild(device-1).setEdgeWeight(edgeWeight[device-1]);
    }

  }

  if(NULL==pS)
  {
    //send the result back to the host
    MPI_Isend((void*)weight, total, MPI_DOUBLE, 0,xWeight, MPI_COMM_WORLD, &req[0]);
    MPI_Isend((void*)edgeWeight, total-1, MPI_DOUBLE, 0,xEdgeWeight, MPI_COMM_WORLD, &req[1]);

    MPI_Waitall(2,req,MPI_STATUSES_IGNORE);
  }

  //clean up
  delete [] weight;
  weight = NULL;
  delete [] edgeWeight;
  edgeWeight = NULL;
  if(pS==NULL)
  {
    delete s;
    s=NULL;
  }
}

/* 
  TODO
  takes a subdomain containing results and copies it into original
  buffer, accounting for invalid ghost zone around edges 
*/
void copy_result_block(DTYPE* buffer, SubDomain3D* s, int pyramidHeight)
{
  
}

void copy_results(DTYPE* buffer, Cluster &cluster, int pyramidHeight)
{
  if(NULL == buffer) return;

  /* get work from all parents and children in cluster */
  for(int n = 0; n < cluster.getNumNodes(); ++n)
  {
    Node &node = cluster.getNode(n);
    int num = node.numSubDomains();
    for(int block =0; block < num; ++block)
    {
      copy_result_block(buffer, node.getSubDomain(block), pyramidHeight);
    }

    for(int c = 0; c < node.getNumChildren(); ++c)
    {
      Node &child = node.getChild(c);
      num = child.numSubDomains();

      for(int block =0; block < num; ++block)
      {
        copy_result_block(buffer, child.getSubDomain(block), pyramidHeight);
      }
    }
  }
}

void runDistributedCell(int rank, int numTasks, DTYPE *data, int x_max, int y_max,
    int z_max, int iterations, int bornMin, int bornMax, 
    int dieMin, int dieMax)

{
  //hack because we want the compiler to give us the 
  //stencil size, but we don't want to have to include
  //the cuda headers in every file, so we convert
  // it to an int array for the time-being.
  dim3 stencil_size(1,1,1);
  int new_stencil_size[3]={stencil_size.z,stencil_size.y,stencil_size.x};
  int deviceCount=0;
  Node myWork;
  Cluster* cluster = NULL;
  struct timeval rec_start, rec_end,comp_start,comp_end,process_start, process_end, balance_start, balance_end;

  myWork.setRank(rank);

  getNumberOfChildren(deviceCount);
  myWork.setNumChildren(deviceCount);
  if(0==rank)
  {
    //get the number of children from other nodes
    cluster = new Cluster(numTasks);
    cluster->getNode(0).setNumChildren(deviceCount);
    receiveNumberOfChildren(numTasks, *cluster);

    /* perform domain decomposition */
    Decomposition decomp;
    int numElements[3] = {z_max,y_max,x_max};
    decomp.decompose(data,3,numElements, new_stencil_size, PYRAMID_HEIGHT);

#ifdef DEBUG
    printDecomposition(decomp);
#endif
    //this is inefficient, need to implement a function that uses Bcast
    for(int node=1; node<cluster->getNumNodes(); ++node)
    {
      benchmarkNode(cluster->getNode(node), decomp.getSubDomain(0));
    }

    benchmarkMyself(cluster->getNode(0),decomp.getSubDomain(0),iterations, bornMin, bornMax, dieMin, dieMax);

    /* now perform the load balancing, assigning task blocks to each node */
    Balancer lb;
    gettimeofday(&balance_start, NULL);                                       
    lb.perfBalance(*cluster,decomp, 0); //passing a 0 means use cpu and gpu on all nodes
    //lb.balance(*cluster,decomp, 0);
    printCluster(*cluster);
    gettimeofday(&balance_end, NULL);                                       
    double balance_sec = ((balance_end.tv_sec - balance_start.tv_sec) +             
        (balance_end.tv_usec - balance_start.tv_usec)/1000000.0);                       
    fprintf(stderr, "***********\nBALANCE TIME: %f seconds.\n",balance_sec);

    gettimeofday(&process_start, NULL);                                       


    //send the work to each node.
    for(int node=1; node < cluster->getNumNodes(); ++node){
      sendData(cluster->getNode(node));
    }
    //root's work is in the first node
    myWork = cluster->getNode(0);
  }
  else
  {
    //send number of children to root
    sendNumberOfChildren(0,deviceCount);
    benchmarkMyself(myWork, NULL,iterations, bornMin, bornMax, dieMin, dieMax);
    receiveData(0, myWork, true, iterations, bornMin, bornMax, dieMin, dieMax);
  }
  if(0==rank)
  {
    /* PROCESS ROOT NODE WORK */
      gettimeofday(&comp_start, NULL);                                       
    for(int task=0; task<myWork.numSubDomains(); ++task){
      processSubDomain(rank,-1, myWork.getSubDomain(task),iterations, bornMin, bornMax, dieMin, dieMax);
    }
    for(int child=0; child<myWork.getNumChildren(); ++child){
      for(int task=0; task<myWork.getChild(child).numSubDomains(); ++task)
      {
        processSubDomain(rank, child, myWork.getChild(child).getSubDomain(task),iterations, bornMin, bornMax, dieMin, dieMax);
      }
    }

      gettimeofday(&comp_end, NULL);                                       
      double time_root_compute= ((comp_end.tv_sec - comp_start.tv_sec) +             
          (comp_end.tv_usec - comp_start.tv_usec)/1000000.0);                       
      fprintf(stderr, "***********\nroot node spent: %f sec processing it's work.\n",time_root_compute);
    if(cluster != NULL) 
    {
      gettimeofday(&rec_start, NULL);                                       
      /* receives results, needs to be asynchronous */
      for(int r=1; r<numTasks; ++r)
      {
        receiveData(r,cluster->getNode(r),false);
      }	
      gettimeofday(&rec_end, NULL);                                       
      gettimeofday(&process_end, NULL);                                       
      double time_root_receive= ((rec_end.tv_sec - rec_start.tv_sec) +             
          (rec_end.tv_usec - rec_start.tv_usec)/1000000.0);                       
      fprintf(stderr, "***********\nroot node spent: %f sec receiving other node's work.\n",time_root_receive);

      double total_sec = ((process_end.tv_sec - process_start.tv_sec) +             
          (process_end.tv_usec - process_start.tv_usec)/1000000.0);                       
      fprintf(stderr, "***********\nTOTAL TIME: %f.\n",total_sec);

    }
  }
  else
  {
    //send my work back to the root
    myWork.setRank(0);
    sendData(myWork);
  }
  if(NULL != cluster)
  {
    delete cluster;
    cluster=NULL;
  }
} 

