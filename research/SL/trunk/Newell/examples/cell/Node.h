#ifndef NODE_H 
#define NODE_H
#include "SubDomain3D.h"
#include <vector>
using namespace std;

class Node{
  double weight;
  int rank;
  vector<SubDomain3D> subD;
  vector<Node> children;

  public: 
    Node();
    Node(double);
    ~Node();
    void addSubDomain(const SubDomain3D&);	
    void setWeight(double);
    void setRank(int);
    const int getRank() const;
    SubDomain3D& getSubDomain(int index) ;	
    const int numSubDomains() const;
    const double getWeight() const;
};

#endif
