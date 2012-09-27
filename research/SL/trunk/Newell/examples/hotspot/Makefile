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
CFLAGS	= $(WARN) $(DEFINES) -DMPICH_IGNORE_CXX_SEEK
LDFLAGS = -fopenmp
OMPFLAGS = $(CFLAGS) -fopenmp
NVFLAGS	=  $(EMU) -arch=sm_20 --compiler-options -Wall  $(F) $(DEFINES)
DEFINES = -DDTYPE=float -DSL_MPI_TYPE=MPI_FLOAT
NVLIBS	= -lcudart -L$(SDK)/C/common/lib -L$(CUDA)/lib64
CXX = g++
MPICXX = mpicxx

all:		distributedHotspot ompHotspot cudaHotspot

ompHotspot:	hotspot-main-OMP.o ompHotspot.o
		$(CXX) -o $@ $^ $(LDFLAGS)

cudaHotspot:	hotspot-main-CUDA.o hotspot.o
		$(CXX) -o $@ $^ $(NVLIBS)

distributedHotspot:	hotspot-main.o distributedHotspot.o hotspot.o ompHotspot.o comm.o Decomposition.o Node.o Cluster.o Balancer.o SubDomain.o
		$(MPICXX) -o $@ $^ $(LDFLAGS) $(NVLIBS)

lint.txt:        ~/cpplint.py hotspot-main.cpp distributedHotspot.cpp ompHotspot.cpp
		~/cpplint.py hotspot-main.cpp distributedHotspot.cpp ompHotspot.cpp > lint.txt

hotspot-main-OMP.o:	hotspot-main-OMP.cpp
		$(CXX) $(CFLAGS) -c $<

hotspot-main-CUDA.o:	hotspot-main-CUDA.cpp
		$(CXX) $(NVINCLUDES) $(CFLAGS) -c $<

hotspot-main.o:	hotspot-main.cpp
		$(MPICXX) $(INCLUDES) $(CFLAGS) -c $<

distributedHotspot.o:	distributedHotspot.cpp
		$(MPICXX) $(NVINCLUDES) $(CFLAGS) -c $<

ompHotspot.o:	ompHotspot.cpp 
		$(CXX) $(INCLUDES) $(OMPFLAGS) -c $<

hotspot.o:	hotspot.cu ../Model.cu ../Model.h
		nvcc $(NVINCLUDES) $(NVFLAGS) -c $<

comm.o:	        ../comm.cpp ../comm.h
		$(MPICXX) $(NVINCLUDES) $(CFLAGS) -c $< 

Decomposition.o:	../Decomposition.cpp ../Decomposition.h ../Node.h
		$(CXX) $(INCLUDES) $(CFLAGS) -c $< 

Balancer.o:	../Balancer.cpp ../Balancer.h ../Cluster.h ../Decomposition.h
		$(CXX) $(INCLUDES) $(CFLAGS) -c $< 

Cluster.o:	../Cluster.cpp ../Cluster.h ../Node.h
		$(CXX) $(INCLUDES) $(CFLAGS) -c $< 

Node.o:		../Node.cpp ../Node.h ../SubDomain.h
		$(CXX) $(INCLUDES) $(CFLAGS) -c $< 

SubDomain.o:	../SubDomain.cpp ../SubDomain.h 
		$(CXX) $(INCLUDES) $(CFLAGS) -c $< 
clean:
		rm -f distributedHotspot ompHotspot cudaHotspot *.o *.linkinfo *~ lint.txt

distclean:	clean
		rm -f hotspot.cu hotspot.h
