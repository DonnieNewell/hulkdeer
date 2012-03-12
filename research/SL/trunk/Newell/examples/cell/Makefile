DBG	= -g -DDEBUG
SDK 	= /localtmp/den4gr/NVIDIA_GPU_Computing_SDK
#EMU	= -deviceemu
CUDA	= /usr/local/cuda
F	= --compiler-options -fno-strict-aliasing
INCLUDES= -I$(SDK)/C/common/inc -I$(CUDA)/include
CFLAGS	= $(INCLUDES) $(DBG)
NVFLAGS	= $(CFLAGS) $(EMU) $(F) -arch=sm_20 
LIBS	= -lcudart -L$(SDK)/C/common/lib -L$(CUDA)/lib64

all:		cell
cell:		cell-main.o cell.o Decomposition.o Node.o SubDomain3D.o
		mpic++ -o $@ $^ $(LDFLAGS) $(LIBS)

cell-main.o:	cell-main.cpp
		mpic++ $(CFLAGS) -c $<

cell.o:		cell.cu ../Model.cu ../Model.h
		nvcc $(NVFLAGS) -c $<

Decomposition.o:	Decomposition.cpp Decomposition.h Node.h
		mpic++ $(CFLAGS) -c $< 

Node.o:		Node.cpp Node.h SubDomain3D.h
		mpic++ $(CFLAGS) -c $< 

SubDomain3D.o:	SubDomain3D.cpp SubDomain3D.h 
		mpic++ $(CFLAGS) -c $< 
clean:
		rm -f cell *.o *.linkinfo *~

distclean:	clean
		rm -f cell.cu cell.h
