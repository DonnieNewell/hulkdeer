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
CFLAGS	= $(WARN) $(DBG) $(DEFINES) -DMPICH_IGNORE_CXX_SEEK
LDFLAGS = -fopenmp
OMPFLAGS = $(CFLAGS) -fopenmp
NVFLAGS	=  $(EMU) $(DBG) -arch=sm_20 --compiler-options -Wall  $(F) $(DEFINES)
DEFINES = -DDTYPE=float -DSL_MPI_TYPE=MPI_FLOAT
NVLIBS	= -lcudart -L$(SDK)/C/common/lib -L$(CUDA)/lib64
CXX = g++
MPICXX = mpicxx

all:		cudaOmpCorrectnessHotspot ompMpiCorrectness

cudaOmpCorrectnessHotspot:	cudaOmpCorrectnessHotspot.o hotspot.o ompHotspot.o
		$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)

ompMpiCorrectness:	ompMpiCorrectness.o comm.o distributedHotspot.o hotspot.o ompHotspot.o Decomposition.o Node.o Cluster.o Balancer.o SubDomain.o
		$(MPICXX) -o $@ $^ -pg $(LDFLAGS) $(LIBS)

cudaOmpCorrectnessHotspot.o:	cudaOmpCorrectnessHotspot.cpp ../ompHotspot.h ../hotspot.h
		$(CXX) $(INCLUDES) $(CFLAGS) -c $<
 
ompMpiCorrectness.o:	ompMpiCorrectness.cpp ../ompHotspot.h ../distributedHotspot.h
		$(MPICXX) $(INCLUDES) $(CFLAGS) -c $<

distributedHotspot.o:	../distributedHotspot.cpp ../distributedHotspot.h
		$(MPICXX) $(INCLUDES) $(CFLAGS) -c $<

ompHotspot.o:	../ompHotspot.cpp ../ompHotspot.h
		$(CXX) $(INCLUDES) $(CFLAGS) -c $<

hotspot.o:		../hotspot.cu ../../Model.cu ../../Model.h
		nvcc $(INCLUDES) $(NVFLAGS) -c $<

comm.o:	        ../../comm.cpp ../../comm.h
		$(MPICXX) $(INCLUDES) $(CFLAGS) -c $< 

Decomposition.o:	../../Decomposition.cpp ../../Decomposition.h ../../Node.h
		$(MPICXX) $(INCLUDES) $(CFLAGS) -c $< 

Balancer.o:	../../Balancer.cpp ../../Balancer.h ../../Cluster.h ../../Decomposition.h
		$(MPICXX) $(INCLUDES) $(CFLAGS) -c $< 

Cluster.o:	../../Cluster.cpp ../../Cluster.h ../../Node.h
		$(MPICXX) $(INCLUDES) $(CFLAGS) -c $< 

Node.o:		../../Node.cpp ../../Node.h ../../SubDomain.h
		$(MPICXX) $(INCLUDES) $(CFLAGS) -c $< 

SubDomain.o:	../../SubDomain.cpp ../../SubDomain.h 
		$(MPICXX) $(INCLUDES) $(CFLAGS) -c $< 
clean:
		rm -f ompMpiCorrectnessHotspot cudaOmpCorrectnessHotspot *.o *.linkinfo *~ lint.txt

distclean:	clean
		rm -f hotspot.cu hotspot.h
