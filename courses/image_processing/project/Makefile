
CC=gcc
NVCC=nvcc
NVFLAGS=-arch sm_13
LIBS = -lnpp -ljpeg
jpeg_sample:jpeg_sample.o
	$(NVCC) $(NVFLAGS) -g -o jpeg_sample jpeg_sample.o $(LIBS)
	
jpeg_sample.o: jpeg_sample.cu
	$(NVCC) $(NVFLAGS) -c -g jpeg_sample.cu -o jpeg_sample.o 

clean:
	rm -rf *.o jpeg_sample
