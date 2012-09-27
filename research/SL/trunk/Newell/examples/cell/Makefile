HOME    = /af21/den4gr
DBG	= -g
SDK 	= $(HOME)/NVIDIA_GPU_Computing_SDK
#EMU	= -deviceemu
#CUDA	= /localtmp/cuda
CUDA	= $(HOME)/cuda
F	= --compiler-options -fno-strict-aliasing
METIS	= $(HOME)/metis-5.0.2/
NVINCLUDES= -I$(SDK)/C/common/inc -I$(CUDA)/include
INCLUDES = -I$(METIS)/include
#MPIFLAGS= -rpath=/af21/den4gr/openmpi-install/lib
WARN    = -Wall -Wextra
DEFINES = -DDTYPE=int -DSL_MPI_TYPE=MPI_INT -DSLAB
CFLAGS	= $(WARN) $(DEFINES) -O3 -DMPICH_IGNORE_CXX_SEEK
LDFLAGS = -fopenmp
CXXLIBS = -L$(METIS)/build/Linux-x86_64/libmetis
OMPFLAGS = $(CFLAGS) -fopenmp
NVFLAGS	=  $(EMU) -arch=sm_20 -O3 --compiler-options -Wall  $(F) $(DEFINES)
NVLIBS	= -lcudart -L$(CUDA)/lib64
MPICXX = mpicxx

all:		distributedCell ompCell cudaCell
 
ompCell:	cell-main-OMP.o ompCell.o
		g++ $(DBG) -o $@ $^ $(LDFLAGS) $(LIBS)

cudaCell:	cell-main-CUDA.o cell.o 
		nvcc $(DBG) -o $@ $^ $(NVFLAGS) $(NVLIBS)

distributedCell:  cell-main.o distributedCell.o cell.o ompCell.o comm.o Decomposition.o Node.o Cluster.o Balancer.o SubDomain.o
		$(MPICXX) $(DBG) -o $@ $^ $(LDFLAGS) $(NVLIBS) $(CXXLIBS)

lint.txt:        ~/cpplint.py cell-main.cpp distributedCell.cpp ompCell.cpp
		~/cpplint.py cell-main.cpp distributedCell.cpp ompCell.cpp > lint.txt 

cell-main-OMP.o:	cell-main-OMP.cpp
		g++ $(DBG) $(CFLAGS) -c $<

cell-main-CUDA.o:	cell-main-CUDA.cpp
		g++ $(NVINCLUDES) $(DBG) $(CFLAGS) -c $<

cell-main.o:	cell-main.cpp
		$(MPICXX) $(INCLUDES) $(DBG) $(CFLAGS) -c $<

distributedCell.o:	distributedCell.cpp
		$(MPICXX) $(NVINCLUDES) $(DBG) $(OMPFLAGS) -c $<

ompCell.o:	ompCell.cpp 
		g++ $(INCLUDES) $(DBG) $(OMPFLAGS) -c $<

cell.o:		cell.cu ../Model.cu ../Model.h
		nvcc $(NVINCLUDES) $(DBG) $(NVFLAGS) -c $<

comm.o:	        ../comm.cpp ../comm.h
		$(MPICXX) $(NVINCLUDES) $(DBG) $(CFLAGS) -c $<

Decomposition.o:	../Decomposition.cpp ../Decomposition.h ../Node.h
		$(MPICXX) $(INCLUDES) $(DBG) $(CFLAGS) -c $<

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
