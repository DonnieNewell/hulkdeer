#DBG	= -pg -DDEBUG
SDK 	= /af21/den4gr/NVIDIA_GPU_Computing_SDK
#EMU	= -deviceemu
CUDA	= /usr/local/cuda
F	= --compiler-options -fno-strict-aliasing
INCLUDES= -I$(SDK)/C/common/inc -I$(CUDA)/include
#MPIFLAGS= -rpath=/af21/den4gr/openmpi-install/lib
LDFLAGS = -fopenmp
CFLAGS	= $(DBG) -Wall -fopenmp
NVFLAGS	= $(DBG)  $(EMU) $(F)
LIBS	= -lcudart -L$(SDK)/C/common/lib -L$(CUDA)/lib64

all:		correctness lint.txt
correctness:	correctness.o distributedCell.o cell.o ompCell.o Decomposition.o Node.o Cluster.o Balancer.o SubDomain3D.o
		mpic++ -o $@ $^ -pg $(LDFLAGS) $(LIBS)
lint.txt:        ~/cpplint.py correctness.cpp 
		~/cpplint.py correctness.cpp > lint.txt 

correctness.o:	correctness.cpp ../ompCell.h ../distributedCell.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $<

distributedCell.o:	../distributedCell.cpp ../distributedCell.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $<

ompCell.o:	../ompCell.cpp ../ompCell.h
		g++ $(INCLUDES) $(CFLAGS) -c $<

cell.o:		../cell.cu ../../Model.cu ../../Model.h
		nvcc $(INCLUDES) $(NVFLAGS) -c $<

Decomposition.o:	../Decomposition.cpp ../Decomposition.h ../Node.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Balancer.o:	../Balancer.cpp ../Balancer.h ../Cluster.h ../Decomposition.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Cluster.o:	../Cluster.cpp ../Cluster.h ../Node.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Node.o:		../Node.cpp ../Node.h ../SubDomain3D.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

SubDomain3D.o:	../SubDomain3D.cpp ../SubDomain3D.h 
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 
clean:
		rm -f correctness *.o *.linkinfo *~ lint.txt

distclean:	clean
		rm -f cell.cu cell.h
