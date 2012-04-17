#ifndef NODE_H 
#define NODE_H
#include "SubDomain3D.h"
#include <vector>
using namespace std;

class Node{
  double weight;
  vector<SubDomain3D> subD;

  public: 
    Node();
    Node(double);
    ~Node();
    void addSubDomain(const SubDomain3D&);	
    void setWeight(double);
    SubDomain3D getSubDomain(int index) const;	
    const int numSubDomains() const;
    const double getWeight() const;
};

#endif
