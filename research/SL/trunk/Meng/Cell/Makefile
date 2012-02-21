CC = /usr/local/cuda/bin/nvcc

#INCLUDE = /usr/local/cuda/include:../common/inc/
LIB = /usr/local/cuda/lib:../common/lib/

INCLUDE = ../common/inc/
LIB = ../common/lib/

LIBRARIES = cudart.lib cutil32.lib

SRC = cell.cu

EXE = cell

release:
	$(CC) $(SRC) -arch sm_12 -o $(EXE) -I$(INCLUDE) -L$(LIB) 

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 

debug:
	$(CC) -arch sm_11 -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 

debugenum:
	$(CC) -g -arch sm_11 -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 
