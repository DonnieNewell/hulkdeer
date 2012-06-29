#ifndef DECOMPOSITION_H 
#define DECOMPOSITION_H
#include "Node.h"
#include <vector>

using namespace std;

class Decomposition{
  vector<SubDomain*> domain;
  void decompose1D(DTYPE*, const int,const int stencil_size[], const int  );
  void decompose2D(DTYPE*, const int, const int,const int stencil_size[], const int);
  void decompose3D(DTYPE*, const int, const int, const int,const int stencil_size[], const int);
  void copyBlock3D(DTYPE* , SubDomain*, const int, const int, const int);
  void copyBlock2D(DTYPE* , SubDomain*, const int, const int);

  public: 
    Decomposition();
    Decomposition(const int);
    Decomposition(const Decomposition&);
    ~Decomposition();
    Decomposition& operator=(const Decomposition&);
    void addSubDomain(SubDomain*);
    const SubDomain* getSubDomain(const int )const;
    SubDomain* getSubDomain(const int );
    SubDomain* popSubDomain();
    size_t getNumSubDomains()const; 
    void decompose(DTYPE* buffer, const int numDimensions, const int numElements[],const int stencil_size[], const int);
};    

void printDecomposition(Decomposition& d);
#endif
