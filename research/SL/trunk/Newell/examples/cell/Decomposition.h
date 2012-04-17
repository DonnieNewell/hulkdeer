#ifndef DECOMPOSITION_H 
#define DECOMPOSITION_H
#include "Node.h"
#include <vector>

using namespace std;

class Decomposition{
  vector<SubDomain3D> domain;
  void decompose1D(const int);
  void decompose2D(const int, const int);
  void decompose3D(const int, const int, const int);

  public: 
    Decomposition();
    Decomposition(const int);
    Decomposition(const Decomposition&);
    ~Decomposition();
    Decomposition& operator=(const Decomposition&);
    void addSubDomain(SubDomain3D&);
    SubDomain3D getSubDomain(const int )const;
    int getNumSubDomains()const; 
    void decompose(const int numDimensions, const int numElements[]);
};

#endif
