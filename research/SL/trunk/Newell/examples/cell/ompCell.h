#ifndef OMP_CELL_H
#define OMP_CELL_H
void runOMPCellSetData(int *, int);
void runOMPCell(int *, int, int, int, int , int bornMin, int bornMax, int dieMin, int dieMax);
void runOMPCellCleanup();
#endif
