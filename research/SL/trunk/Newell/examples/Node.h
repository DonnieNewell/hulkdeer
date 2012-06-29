#ifndef NODE_H
#define NODE_H
#include "SubDomain.h"
#include <vector>
#include <queue>
#include <utility>

using namespace std;

class WorkRequest {
  public:
    WorkRequest() {
      req.first  = 0.0;
      req.second = 0  ;
    }

    WorkRequest(const WorkRequest& rhs) {
      if (this != &rhs) {
        req.first   = rhs.getTimeDiff();
        req.second  = rhs.getIndex();
      }
    }

    WorkRequest(double timeDiff, int index ) {
      req.first  = timeDiff;
      req.second = index;
    }

    ~WorkRequest() { }
    bool operator<(const WorkRequest& rhs) const {
      return req.first < rhs.getTimeDiff();
    }

    bool operator>(const WorkRequest& rhs) const {
      return req.first > rhs.getTimeDiff();
    }

    double getTimeDiff() const { return req.first; }
    double setTimeDiff(double newTimeDiff) { return req.first = newTimeDiff; }
    int getIndex() const { return req.second; }
    int setIndex(int newIndex) { return req.second = newIndex; }
  private:
    pair<double,int> req;
};

typedef priority_queue< WorkRequest, vector<WorkRequest>, greater<WorkRequest> > WorkQueue;
/************************************************/

class Node{
  double weight;
  double edgeWeight;
  int rank;
  vector<SubDomain*> subD;
  vector<Node> children;

  public:
  Node();
  Node(double);
  Node(const Node&);
  ~Node();
  Node& operator=(const Node& rhs);
  void addSubDomain(SubDomain*);
  void setEdgeWeight(double);
  void setWeight(double);
  void setRank(int);
  void setNumChildren(int);
  const unsigned int getNumChildren() const;
  const int getRank() const;
  const double getTimeEst(int extra) const;
  int getWorkNeeded(const double runtime) const;
  int getTotalWorkNeeded(const double runtime) const;
  const unsigned int numTotalSubDomains() const;
  SubDomain* getSubDomain(int index) const;
  SubDomain* globalGetSubDomain(int index) const;
  SubDomain* getSubDomainLinear(int index) const;
  SubDomain* popSubDomain() ;
  Node& getChild(int index);
  const Node& getChild(int index) const;
  const unsigned int numSubDomains() const;
  const double getWeight() const;
  const double getEdgeWeight() const;
  const double getTotalWeight() const;
  const double getMinEdgeWeight() const;
};

#endif
