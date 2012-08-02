DBG	= -g -pg
SDK 	= /af21/den4gr/NVIDIA_GPU_Computing_SDK
#EMU	= -deviceemu
CUDA	= /usr/local/cuda
F	= --compiler-options -fno-strict-aliasing
INCLUDES= -I$(SDK)/C/common/inc -I$(CUDA)/include
#MPIFLAGS= -rpath=/af21/den4gr/openmpi-install/lib
LDFLAGS = -fopenmp
DEFINES = -DSL_MPI_TYPE=MPI_INT -DDTYPE=int
CFLAGS	= $(DBG) -Wall -fopenmp $(DEFINES)
NVFLAGS	= $(DBG)  $(EMU) $(F) -arch sm_20 $(DEFINES)
LIBS	= -lcudart -L$(SDK)/C/common/lib -L$(CUDA)/lib64 

all:		cudaOmpCorrectness ompMpiCorrectness lint.txt

cudaOmpCorrectness:	cudaOmpCorrectness.o cell.o ompCell.o
		g++ -o $@ $^ -pg $(LDFLAGS) $(LIBS)

ompMpiCorrectness:	ompMpiCorrectness.o comm.o distributedCell.o cell.o ompCell.o Decomposition.o Node.o Cluster.o Balancer.o SubDomain.o
		mpic++ -o $@ $^ -pg $(LDFLAGS) $(LIBS)

lint.txt:        ~/cpplint.py ompMpiCorrectness.cpp 
		~/cpplint.py ompMpiCorrectness.cpp > lint.txt 

cudaOmpCorrectness.o:	cudaOmpCorrectness.cpp ../ompCell.h ../cell.h
		g++ $(INCLUDES) $(CFLAGS) -c $<

ompMpiCorrectness.o:	ompMpiCorrectness.cpp ../ompCell.h ../distributedCell.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $<

distributedCell.o:	../distributedCell.cpp ../distributedCell.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $<

ompCell.o:	../ompCell.cpp ../ompCell.h
		g++ $(INCLUDES) $(CFLAGS) -c $<

cell.o:		../cell.cu ../../Model.cu ../../Model.h
		nvcc $(INCLUDES) $(NVFLAGS) -c $<

comm.o:	        ../../comm.cpp ../../comm.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Decomposition.o:	../../Decomposition.cpp ../../Decomposition.h ../../Node.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Balancer.o:	../../Balancer.cpp ../../Balancer.h ../../Cluster.h ../../Decomposition.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Cluster.o:	../../Cluster.cpp ../../Cluster.h ../../Node.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Node.o:		../../Node.cpp ../../Node.h ../../SubDomain.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

SubDomain.o:	../../SubDomain.cpp ../../SubDomain.h 
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 
clean:
		rm -f ompMpiCorrectness cudaOmpCorrectness *.o *.linkinfo *~ lint.txt

distclean:	clean
		rm -f cell.cu cell.h
