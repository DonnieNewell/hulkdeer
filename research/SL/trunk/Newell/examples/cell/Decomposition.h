#ifndef DECOMPOSITION_H 
#define DECOMPOSITION_H
#include "Node.h"
#include <vector>

using namespace std;

class Decomposition{
  vector<SubDomain3D> domain;
  void decompose1D(int*, const int);
  void decompose2D(int*, const int, const int);
  void decompose3D(int*, const int, const int, const int);
  void copyBlock(int* , SubDomain3D& , const int ,const int ,const int );

  public: 
    Decomposition();
    Decomposition(const int);
    Decomposition(const Decomposition&);
    ~Decomposition();
    Decomposition& operator=(const Decomposition&);
    void addSubDomain(SubDomain3D&);
    const SubDomain3D& getSubDomain(const int )const;
    SubDomain3D& getSubDomain(const int );
    int getNumSubDomains()const; 
    void decompose(int* buffer, const int numDimensions, const int numElements[]);
};    

#endif
