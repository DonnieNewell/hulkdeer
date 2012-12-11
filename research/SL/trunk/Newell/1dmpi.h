#ifndef runCell_H
#define runCell_H
#include <mpi.h>
#define SL_MPI_Init() int _size_, _rank_; 
MPI_Init(&argc, &argv); 
MPI_Comm_size(MPI_COMM_WORLD, &_size_); 
MPI_Comm_rank(MPI_COMM_WORLD, &_rank_)
#define SL_MPI_Finalize() MPI_Finalize()
void runCellSetData(int *, int);
void runCell(int *, #endifint, int , int bornMin, int bornMax, int dieMin, int dieMax);
