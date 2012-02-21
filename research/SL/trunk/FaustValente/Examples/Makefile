DBG	= -g
#EMU	= -deviceemu
F	= --compiler-options -fno-strict-aliasing
CFLAGS	= $(DBG) $(EMU) $(F)
LIBS	= -lcudart

all:		plate hotspot pathfinder cell

plate:		plate-main.o plate.o
		nvcc -o $@ plate-main.o plate.o $(LDFLAGS) $(LIBS)

plate-main.o:	plate-main.cpp
		nvcc $(CFLAGS) -c $<

plate.o:	plate.cu Model.cu Model.h
		nvcc $(CFLAGS) -c $<

hotspot:	hotspot-main.o hotspot.o
		nvcc -o $@ hotspot-main.o hotspot.o $(LDFLAGS) $(LIBS)

hotspot-main.o:	hotspot-main.cpp
		nvcc $(CFLAGS) -c $<

hotspot.o:	hotspot.cu Model.cu Model.h
		nvcc $(CFLAGS) -c $<

pathfinder:	pathfinder-main.o pathfinder.o
		nvcc -o $@ pathfinder-main.o pathfinder.o $(LDFLAGS) $(LIBS)

pathfinder-main.o:	pathfinder-main.cpp
		nvcc $(CFLAGS) -c $<

pathfinder.o:	pathfinder.cu Model.cu Model.h
		nvcc $(CFLAGS) -c $<

cell:		cell-main.o cell.o
		nvcc -o $@ cell-main.o cell.o $(LDFLAGS) $(LIBS)

cell-main.o:	cell-main.cpp
		nvcc $(CFLAGS) -c $<

cell.o:		cell.cu Model.cu Model.h
		nvcc $(CFLAGS) -c $<

clean:
		rm -f plate hotspot pathfinder cell *.o *.linkinfo *~

distclean:	clean
		rm -f plate.cu plate.h hotspot.cu hotspot.h pathfinder.cu pathfinder.h cell.cu cell.h
