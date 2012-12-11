cmake_minimum_required(VERSION 2.6)
project(cell_test)
 
####################################
# GTest
####################################
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

####################################
# OpenMP
####################################
find_package(OpenMP REQUIRED)
 
####################################
# CUDA
####################################
find_package(CUDA REQUIRED)
include(FindCUDA) 
include_directories(/if10/den4gr/NVIDIA_GPU_Computing_SDK/C/common/inc)

####################################
# MPI
####################################
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

####################################
# SL
####################################
add_definitions(-DDTYPE=int)
add_definitions(-DSL_MPI_TYPE=MPI_INT)

####################################
# Unit Tests
####################################

add_library(subdomain ../../SubDomain.cpp)
add_library(node ../../Node.cpp)
target_link_libraries(node subdomain)

add_library(decomposition ../../Decomposition.cpp)
target_link_libraries(decomposition node)

add_library(cluster ../../Cluster.cpp)
target_link_libraries(cluster decomposition)

add_library(balancer ../../Balancer.cpp)
target_link_libraries(balancer cluster)

add_library(comm ../../comm.cpp)
target_link_libraries(comm cluster mpi)

add_library(ompCell ../ompCell.cpp)
target_link_libraries(ompCell comm)

cuda_add_library(cell ../cell.cu ../../Model.cu)
#set_target_properties(cell PROPERTIES LINKER_LANGUAGE C)

cuda_add_library(distributedCell ../distributedCell.cpp)
target_link_libraries(distributedCell ompCell cell balancer)

add_executable(ompMpiCorrectness ompMpiCorrectness.cpp)
target_link_libraries(ompMpiCorrectness ompCell distributedCell ${GTEST_LIBRARIES} pthread)
if(MPI_COMPILE_FLAGS)
  set_target_properties(ompMpiCorrectness PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(ompMpiCorrectness PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
add_test(ompMpiCorrectness ompMpiCorrectness)

add_executable(cudaOmpCorrectness cudaOmpCorrectness.cpp)
target_link_libraries(cudaOmpCorrectness ompCell cell ${GTEST_LIBRARIES} pthread)
add_test(cudaOmpCorrectness cudaOmpCorrectness)

add_executable(outerInnerCorrectness outerInnerCorrectness.cpp)
target_link_libraries(outerInnerCorrectness ompCell ${GTEST_LIBRARIES} pthread)
add_test(outerInnerCorrectness outerInnerCorrectness)
