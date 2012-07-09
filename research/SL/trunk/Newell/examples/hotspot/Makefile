#DBG	= -DDEBUG
DBG	= -g
SDK 	= /af21/den4gr/NVIDIA_GPU_Computing_SDK
#EMU	= -deviceemu
CUDA	= /usr/local/cuda
F	= --compiler-options -fno-strict-aliasing
INCLUDES= -I$(SDK)/C/common/inc -I$(CUDA)/include
#MPIFLAGS= -rpath=/af21/den4gr/openmpi-install/lib
LDFLAGS = -fopenmp
DEFINES = -DDTYPE=float -DSL_MPI_TYPE=MPI_FLOAT
CFLAGS	= $(DBG) -Wall -fopenmp $(DEFINES)
NVFLAGS	=  $(EMU) $(F) -arch sm_20 $(DEFINES)
LIBS	= -lcudart -L$(SDK)/C/common/lib -L$(CUDA)/lib64

all:		hotspot
hotspot:	hotspot-main.o comm.o distributedHotspot.o hotspot.o ompHotspot.o Decomposition.o Node.o Cluster.o Balancer.o SubDomain.o
		mpic++ -o $@ $^ $(LDFLAGS) $(LIBS)
lint.txt:        ~/cpplint.py hotspot-main.cpp distributedHotspot.cpp ompHotspot.cpp
		~/cpplint.py hotspot-main.cpp distributedHotspot.cpp ompHotspot.cpp > lint.txt 

hotspot-main.o:	hotspot-main.cpp
		mpic++ $(INCLUDES) $(CFLAGS) -c $<

distributedHotspot.o:	distributedHotspot.cpp
		mpic++ $(INCLUDES) $(CFLAGS) -c $<

ompHotspot.o:	ompHotspot.cpp 
		g++ $(INCLUDES) $(CFLAGS) -c $<

hotspot.o:	hotspot.cu ../Model.cu ../Model.h
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
		rm -f hotspot *.o *.linkinfo *~ lint.txt

distclean:	clean
		rm -f hotspot.cu hotspot.h
