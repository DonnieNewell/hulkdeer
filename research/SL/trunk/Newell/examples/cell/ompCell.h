#ifndef OMP_CELL_H
#define OMP_CELL_H
void runOMPCellSetData(int *, int);
void runOMPCell(DTYPE*, int, int, int, int, const int kPyramidHeight,
                int bornMin, int bornMax, int dieMin, int dieMax);
void runOMPCellOuter(DTYPE*, int, int, int, int, const int kPyramidHeight,
                int bornMin, int bornMax, int dieMin, int dieMax);
void runOMPCellInner(DTYPE*, int, int, int, int, const int kPyramidHeight,
                int bornMin, int bornMax, int dieMin, int dieMax);
void runOMPCellCleanup();
#endif
