DBG	= -g
SDK 	= /localtmp/den4gr/NVIDIA_GPU_Computing_SDK
#EMU	= -deviceemu
F	= --compiler-options -fno-strict-aliasing
CFLAGS	= $(EMU) $(F) -arch=sm_20 -I$(SDK)/C/common/inc
LIBS	= -lcudart -L$(SDK)/C/common/lib

all:		cell
cell:		cell-main.o cell.o
		nvcc -o $@ cell-main.o cell.o $(LDFLAGS) $(LIBS)

cell-main.o:	cell-main.cpp
		nvcc $(CFLAGS) -c $<

cell.o:		cell.cu Model.cu Model.h
		nvcc $(CFLAGS) -c $<

clean:
		rm -f cell *.o *.linkinfo *~

distclean:	clean
		rm -f cell.cu cell.h
