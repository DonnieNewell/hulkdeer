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
    Node(const Node&);
    ~Node();
    Node& operator=(const Node& rhs);
    void addSubDomain(const SubDomain3D&);	
    void setWeight(double);
    void setRank(int);
    void setNumChildren(int);
    const int getNumChildren() const;
    const int getRank() const;
    SubDomain3D& getSubDomain(int index) ;	
    Node& getChild(int index) ;	
    const int numSubDomains() const;
    const double getWeight() const;
};

#endif
