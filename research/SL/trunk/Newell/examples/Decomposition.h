#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H
#include "Node.h"
#include <deque>

using namespace std;

class Decomposition{
  deque<SubDomain*> domain;
  void decompose2D(DTYPE*, const int, const int,const int stencil_size[],
                    const int, const int);
  void decompose3D(DTYPE*, const int, const int, const int, const int [],
                    const int, const int);
  void decompose3DSlab(DTYPE*, const int, const int, const int, const int [],
                    const int, const int);
  void copyBlock3D(DTYPE* , SubDomain*, const int, const int, const int);
  void copyBlock2D(DTYPE* , SubDomain*, const int, const int);

  public:
    Decomposition();
    Decomposition(const Decomposition&);
    ~Decomposition();
    Decomposition& operator=(const Decomposition&);
    void addSubDomain(SubDomain*);
    const SubDomain* getSubDomain(const int )const;
    SubDomain* getSubDomain(const int);
    SubDomain* getAggregate3D(const unsigned int kNumBlocksToCombine);
    SubDomain* popSubDomain();
    size_t getNumSubDomains()const;
    void decompose(DTYPE* , const int , const int [],const int [], const int,
                    const int);
};

void printDecomposition(Decomposition& d);
#endif
