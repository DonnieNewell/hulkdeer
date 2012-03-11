#ifndef NODE_H 
#define NODE_H
#include "SubDomain3D.h"

class Node{
  double weight;
  SubDomain3D subD;

  public: 
    Node();
    Node(double,const SubDomain3D&);
    ~Node();
    void setSubDomain(const SubDomain3D&);	
    void setWeight(double);
    SubDomain3D getSubDomain() const;	
    double getWeight();
};

#endif
