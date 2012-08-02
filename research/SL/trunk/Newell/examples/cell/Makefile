DBG	= -g
SDK 	= /af21/den4gr/NVIDIA_GPU_Computing_SDK
#EMU	= -deviceemu
CUDA	= /usr/local/cuda
F	= --compiler-options -fno-strict-aliasing
NVINCLUDES= -I$(SDK)/C/common/inc -I$(CUDA)/include
#MPIFLAGS= -rpath=/af21/den4gr/openmpi-install/lib
WARN    = -Wall -Wextra
DEFINES = -DDTYPE=int -DSL_MPI_TYPE=MPI_INT
CFLAGS	= $(WARN) -O2 $(DEFINES)
LDFLAGS = -fopenmp
OMPFLAGS = $(CFLAGS) -fopenmp
NVFLAGS	=  $(EMU) -arch=sm_20 --compiler-options -Wall -O2 $(F) $(DEFINES)
NVLIBS	= -lcudart -L$(SDK)/C/common/lib -L$(CUDA)/lib64

all:		distributedCell ompCell cudaCell
 
ompCell:	cell-main-OMP.o ompCell.o
		g++ $(DBG) -o $@ $^ $(LDFLAGS) $(LIBS)

cudaCell:	cell-main-CUDA.o cell.o 
		nvcc $(DBG) -o $@ $^ $(NVFLAGS) $(NVLIBS)

distributedCell:  cell-main.o distributedCell.o cell.o ompCell.o comm.o Decomposition.o Node.o Cluster.o Balancer.o SubDomain.o
		mpic++ $(DBG) -o $@ $^ $(LDFLAGS) $(NVLIBS)

lint.txt:        ~/cpplint.py cell-main.cpp distributedCell.cpp ompCell.cpp
		~/cpplint.py cell-main.cpp distributedCell.cpp ompCell.cpp > lint.txt 

cell-main-OMP.o:	cell-main-OMP.cpp
		g++ $(DBG) $(CFLAGS) -c $<

cell-main-CUDA.o:	cell-main-CUDA.cpp
		g++ $(NVINCLUDES) $(DBG) $(CFLAGS) -c $<

cell-main.o:	cell-main.cpp
		mpic++ $(INCLUDES) $(DBG) $(CFLAGS) -c $<

distributedCell.o:	distributedCell.cpp
		mpic++ $(NVINCLUDES) $(DBG) $(CFLAGS) -c $<

ompCell.o:	ompCell.cpp 
		g++ $(INCLUDES) $(DBG) $(OMPFLAGS) -c $<

cell.o:		cell.cu ../Model.cu ../Model.h
		nvcc $(NVINCLUDES) $(DBG) $(NVFLAGS) -c $<

comm.o:	        ../comm.cpp ../comm.h
		mpic++ $(NVINCLUDES) $(DBG) $(CFLAGS) -c $<

Decomposition.o:	../Decomposition.cpp ../Decomposition.h ../Node.h
		mpic++ $(INCLUDES) $(DBG) $(CFLAGS) -c $<

Balancer.o:	../Balancer.cpp ../Balancer.h ../Cluster.h ../Decomposition.h
		g++ $(INCLUDES) $(DBG) $(CFLAGS) -c $<

Cluster.o:	../Cluster.cpp ../Cluster.h ../Node.h
		g++ $(INCLUDES) $(DBG) $(CFLAGS) -c $<

Node.o:		../Node.cpp ../Node.h ../SubDomain.h
		g++ $(INCLUDES) $(DBG) $(CFLAGS) -c $<

SubDomain.o:	../SubDomain.cpp ../SubDomain.h 
		g++ $(INCLUDES) $(DBG) $(CFLAGS) -c $<
clean:
		rm -f distributedCell ompCell cudaCell gmon.* *.o *.linkinfo *~ lint.txt

distclean:	clean
		rm -f cell.cu cell.h
