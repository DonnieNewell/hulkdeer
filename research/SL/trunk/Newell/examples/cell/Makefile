#DBG	= -pg -DDEBUG
SDK 	= /af21/den4gr/NVIDIA_GPU_Computing_SDK
#EMU	= -deviceemu
CUDA	= /usr/local/cuda
F	= --compiler-options -fno-strict-aliasing
INCLUDES= -I$(SDK)/C/common/inc -I$(CUDA)/include
#MPIFLAGS= -rpath=/af21/den4gr/openmpi-install/lib
LDFLAGS = -fopenmp
DEFINES = -DDTYPE=int -DSL_MPI_TYPE=MPI_INT
CFLAGS	= $(DBG) -Wall -fopenmp $(DEFINES)
NVFLAGS	= $(DBG)  $(EMU) $(F) -arch sm_20 $(DEFINES)
LIBS	= -lcudart -L$(SDK)/C/common/lib -L$(CUDA)/lib64

all:		cell 
cell:		cell-main.o comm.o distributedCell.o cell.o ompCell.o Decomposition.o Node.o Cluster.o Balancer.o SubDomain.o
		mpic++ -o $@ $^ $(LDFLAGS) $(LIBS)
lint.txt:        ~/cpplint.py cell-main.cpp distributedCell.cpp ompCell.cpp
		~/cpplint.py cell-main.cpp distributedCell.cpp ompCell.cpp > lint.txt 

cell-main.o:	cell-main.cpp
		mpic++ $(INCLUDES) $(CFLAGS) -c $<

distributedCell.o:	distributedCell.cpp
		mpic++ $(INCLUDES) $(CFLAGS) -c $<

ompCell.o:	ompCell.cpp 
		g++ $(INCLUDES) $(CFLAGS) -c $<

cell.o:		cell.cu ../Model.cu ../Model.h
		nvcc $(INCLUDES) $(NVFLAGS) -c $<

comm.o:	        ../comm.cpp ../comm.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Decomposition.o:	../Decomposition.cpp ../Decomposition.h ../Node.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Balancer.o:	../Balancer.cpp ../Balancer.h ../Cluster.h ../Decomposition.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Cluster.o:	../Cluster.cpp ../Cluster.h ../Node.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Node.o:		../Node.cpp ../Node.h ../SubDomain.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

SubDomain.o:	../SubDomain.cpp ../SubDomain.h 
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 
clean:
		rm -f cell *.o *.linkinfo *~ lint.txt

distclean:	clean
		rm -f cell.cu cell.h
