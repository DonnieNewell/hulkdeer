#ifndef CELL_H
#define CELL_H
void runCellSetData(DTYPE *host_data, int num_elements);
void runCell(DTYPE *host_data, int x_max, int y_max, int z_max, int iterations,
        const int kPyramidHeight, int bornMin, int bornMax, int dieMin,
        int dieMax, int device);
void runCellInner(DTYPE *host_data, int x_max, int y_max, int z_max, int iterations,
        const int kPyramidHeight, int bornMin, int bornMax, int dieMin,
        int dieMax, int device);
void runCellOuter(DTYPE *host_data, int x_max, int y_max, int z_max, int iterations,
        const int kPyramidHeight, int bornMin, int bornMax, int dieMin,
        int dieMax, int device);
void runCellCleanup();
double getGpuMemcpyTime();
#endif
