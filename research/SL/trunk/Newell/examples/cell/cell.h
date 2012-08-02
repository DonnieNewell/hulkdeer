#ifndef CELL_H
#define CELL_H
void runCellSetData(int *, int);
void runCell(DTYPE *, int, int, int, int, const int kPyramidHeight, int bornMin,
        int bornMax, int dieMin, int dieMax, int device);
void runCellCleanup();

#endif