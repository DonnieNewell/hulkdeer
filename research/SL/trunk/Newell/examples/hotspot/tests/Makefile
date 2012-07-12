DBG	= -g 
SDK 	= /af21/den4gr/NVIDIA_GPU_Computing_SDK
#EMU	= -deviceemu
CUDA	= /usr/local/cuda
F	= --compiler-options -fno-strict-aliasing
INCLUDES= -I$(SDK)/C/common/inc -I$(CUDA)/include
#MPIFLAGS= -rpath=/af21/den4gr/openmpi-install/lib
LDFLAGS = -fopenmp
DEFINES = -DSL_MPI_TYPE=MPI_FLOAT -DDTYPE=float
CFLAGS	= $(DBG) -Wall -fopenmp $(DEFINES)
NVFLAGS	= $(DBG)  $(EMU) $(F) -arch sm_20 $(DEFINES)
LIBS	= -lcudart -L$(SDK)/C/common/lib -L$(CUDA)/lib64 

all:		cudaOmpCorrectnessHotspot

cudaOmpCorrectnessHotspot:	cudaOmpCorrectnessHotspot.o hotspot.o ompHotspot.o
		g++ -o $@ $^ -pg $(LDFLAGS) $(LIBS)

cudaOmpCorrectnessHotspot.o:	cudaOmpCorrectnessHotspot.cpp ../ompHotspot.h ../hotspot.h
		g++ $(INCLUDES) $(CFLAGS) -c $<

ompHotspot.o:	../ompHotspot.cpp ../ompHotspot.h
		g++ $(INCLUDES) $(CFLAGS) -c $<

hotspot.o:		../hotspot.cu ../../Model.cu ../../Model.h
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
		rm -f ompMpiCorrectnessHotspot cudaOmpCorrectnessHotspot *.o *.linkinfo *~ lint.txt

distclean:	clean
		rm -f hotspot.cu hotspot.h
