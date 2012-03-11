#ifndef DECOMPOSITION_H 
#define DECOMPOSITION_H
#include "Node.h"
#include <vector>

using namespace std;

class Decomposition{
  vector<Node> domain;
  void decompose1D(const int);
  void decompose2D(const int, const int);
  void decompose3D(const int, const int, const int);

  public: 
    Decomposition();
    Decomposition(const int);
    Decomposition(const Decomposition&);
    ~Decomposition();
    Decomposition& operator=(const Decomposition&);
    void addNode(Node&);
    Node getNode(const int )const;
    int getNumNodes()const; 
    void decompose(const int numDimensions, const int numElements[]);
    void normalize();	
};

#endif
