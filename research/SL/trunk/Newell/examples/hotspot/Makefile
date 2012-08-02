#DBG	= -DDEBUG
DBG	= -g
SDK 	= /af21/den4gr/NVIDIA_GPU_Computing_SDK
#EMU	= -deviceemu
CUDA	= /usr/local/cuda
F	= --compiler-options -fno-strict-aliasing
NVINCLUDES= -I$(SDK)/C/common/inc -I$(CUDA)/include
#MPIFLAGS= -rpath=/af21/den4gr/openmpi-install/lib
WARN    = -Wall -Wextra
DEFINES = -DDTYPE=float -DSL_MPI_TYPE=MPI_FLOAT
CFLAGS	= $(DBG) $(WARN) -O2 $(DEFINES)
LDFLAGS = -fopenmp
OMPFLAGS = $(CFLAGS) $(LDFLAGS)
NVFLAGS	=  $(EMU) $(DBG) -arch=sm_20 --compiler-options -Wall -O2 $(F) $(DEFINES)
NVLIBS	= -lcudart -L$(SDK)/C/common/lib -L$(CUDA)/lib64

all:		distributedHotspot ompHotspot cudaHotspot

ompHotspot:	hotspot-main-OMP.o ompHotspot.o
		g++ -o $@ $^ $(LDFLAGS)

cudaHotspot:	hotspot-main-CUDA.o hotspot.o
		g++ -o $@ $^ $(NVLIBS)

distributedHotspot:	hotspot-main.o distributedHotspot.o hotspot.o ompHotspot.o comm.o Decomposition.o Node.o Cluster.o Balancer.o SubDomain.o
		mpic++ -o $@ $^ $(LDFLAGS) $(NVLIBS)

lint.txt:        ~/cpplint.py hotspot-main.cpp distributedHotspot.cpp ompHotspot.cpp
		~/cpplint.py hotspot-main.cpp distributedHotspot.cpp ompHotspot.cpp > lint.txt

hotspot-main-OMP.o:	hotspot-main-OMP.cpp
		g++ $(CFLAGS) -c $<

hotspot-main-CUDA.o:	hotspot-main-CUDA.cpp
		g++ $(NVINCLUDES) $(CFLAGS) -c $<

hotspot-main.o:	hotspot-main.cpp
		mpic++ $(INCLUDES) $(CFLAGS) -c $<

distributedHotspot.o:	distributedHotspot.cpp
		mpic++ $(NVINCLUDES) $(CFLAGS) -c $<

ompHotspot.o:	ompHotspot.cpp 
		g++ $(INCLUDES) $(OMPFLAGS) -c $<

hotspot.o:	hotspot.cu ../Model.cu ../Model.h
		nvcc $(NVINCLUDES) $(NVFLAGS) -c $<

comm.o:	        ../comm.cpp ../comm.h
		mpic++ $(NVINCLUDES) $(CFLAGS) -c $< 

Decomposition.o:	../Decomposition.cpp ../Decomposition.h ../Node.h
		g++ $(INCLUDES) $(CFLAGS) -c $< 

Balancer.o:	../Balancer.cpp ../Balancer.h ../Cluster.h ../Decomposition.h
		g++ $(INCLUDES) $(CFLAGS) -c $< 

Cluster.o:	../Cluster.cpp ../Cluster.h ../Node.h
		g++ $(INCLUDES) $(CFLAGS) -c $< 

Node.o:		../Node.cpp ../Node.h ../SubDomain.h
		g++ $(INCLUDES) $(CFLAGS) -c $< 

SubDomain.o:	../SubDomain.cpp ../SubDomain.h 
		g++ $(INCLUDES) $(CFLAGS) -c $< 
clean:
		rm -f distributedHotspot ompHotspot cudaHotspot *.o *.linkinfo *~ lint.txt

distclean:	clean
		rm -f hotspot.cu hotspot.h
