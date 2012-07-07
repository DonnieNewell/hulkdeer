DBG	= -g 
SDK 	= /af21/den4gr/NVIDIA_GPU_Computing_SDK
#EMU	= -deviceemu
CUDA	= /usr/local/cuda
F	= --compiler-options -fno-strict-aliasing
INCLUDES= -I$(SDK)/C/common/inc -I$(CUDA)/include
#MPIFLAGS= -rpath=/af21/den4gr/openmpi-install/lib
DEFINES = -DSL_MPI_TYPE=MPI_INT -DDTYPE=int
CFLAGS	= $(DBG) -Wall -fopenmp $(DEFINES)
LIBS	= -lcudart -L$(SDK)/C/common/lib -L$(CUDA)/lib64 

all:		commTest

commTest:	commTest.o comm.o Cluster.o Node.o SubDomain.o
		mpic++ -o $@ $^ $(LDFLAGS) $(LIBS)

commTest.o:	commTest.cpp ../comm.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $<

comm.o:	        ../comm.cpp ../comm.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Cluster.o:	../Cluster.cpp ../Cluster.h 
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

Node.o:		../Node.cpp ../Node.h ../SubDomain.h
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

SubDomain.o:	../SubDomain.cpp ../SubDomain.h 
		mpic++ $(INCLUDES) $(CFLAGS) -c $< 

clean:
		rm -f commTest *.o *.linkinfo *~ lint.txt
