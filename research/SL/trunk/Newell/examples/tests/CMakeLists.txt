cmake_minimum_required(VERSION 2.6)
project(common_test)
 
####################################
# GTest
####################################
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

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
add_library(subdomain ../SubDomain.cpp)
add_library(node ../Node.cpp)
target_link_libraries(node subdomain)

add_library(decomposition ../Decomposition.cpp)
target_link_libraries(decomposition node)

add_library(cluster ../Cluster.cpp)
target_link_libraries(cluster decomposition)

add_library(balancer ../Balancer.cpp)
target_link_libraries(balancer cluster)

add_library(comm ../comm.cpp)
target_link_libraries(comm cluster mpi)

add_executable(comm2DTest comm2DTest.cpp)
add_executable(comm3DTest comm3DTest.cpp)
target_link_libraries(comm2DTest comm ${GTEST_LIBRARIES} pthread)
target_link_libraries(comm3DTest comm ${GTEST_LIBRARIES} pthread)
if(MPI_COMPILE_FLAGS)
  set_target_properties(comm2DTest PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(comm3DTest PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(comm2DTest PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
  set_target_properties(comm3DTest PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
add_test(comm comm2DTest)
add_test(comm comm3DTest)

add_executable(aggregate aggregate.cpp)
target_link_libraries(aggregate decomposition ${GTEST_LIBRARIES} pthread)
add_test(aggregate aggregate)

add_executable(balanceTest balanceTest.cpp)
target_link_libraries(balanceTest balancer ${GTEST_LIBRARIES} pthread)
add_test(balanceTest balanceTest)
