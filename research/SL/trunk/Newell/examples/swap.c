#include <stdio.h>
#include <cstdlib>

#define SWAP(first, second) ((a^=b),(b^=a),(a^=b))

int main(int argc, char** argv) {
if (argc != 3) return -1;

int a = atoi(argv[1]);
int b = atoi(argv[2]);
printf("a:%d , b:%d\n", a, b);

printf("SWAP!\n");
SWAP(a, b);

printf("a:%d , b:%d\n", a, b);

printf("determine log2(%d).\n", a);
unsigned int count = 0;
unsigned int temp = static_cast<unsigned int>(a);
while (temp >>= 1) ++count;

printf("log2(%d) = %d.\n", a, count);
return 0;
}
