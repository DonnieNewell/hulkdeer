CC = /usr/local/cuda/bin/nvcc

#INCLUDE = /usr/local/cuda/include:../common/inc/
LIB = /usr/local/cuda/lib:../common/lib/

INCLUDE = ../common/inc/
LIB = ../common/lib/

LIBRARIES = cudart.lib cutil32.lib

SRC = dynproc.cu

EXE = dynproc

release:
	$(CC) $(SRC) -arch sm_11 -o $(EXE) -I$(INCLUDE) -L$(LIB) 
	#$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 

debug:
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 

debugenum:
	$(CC) -g -arch sm_11 -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 
	#$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 


#export CUDA_PROFILE=1
#export CUDA_PROFILE_CONFIG=/af18/jm6dg/cuda_sdk/cuda_profile.config 
#export CUDA_PROFILE_CONFIG=/af18/jm6dg/cuda_sdk/cuda_profile_mem.config 


