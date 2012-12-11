#ifndef NODE_H
#define NODE_H
#include "SubDomain.h"
#include <vector>
#include <map>
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
void printWorkQueue(WorkQueue& queue);
/************************************************/

class Node{
  double weight;
  double edgeWeight;
  int rank;
  int balance_count;
  bool is_CPU;
  vector<SubDomain*> subD;
  vector<Node> children;
  map<int,int> linear_lookup;

  public:
  Node();
  Node(double);
  Node(const Node&);
  ~Node();
  Node& operator=(const Node&);
  void addSubDomain(SubDomain*);
  void setEdgeWeight(double);
  void setWeight(double);
  void setRank(int);
  void incrementBalCount();
  void decrementBalCount();
  int getBalCount() const;
  void setCPU(const bool);
  bool isCPU() const;
  void setNumChildren(const int);
  unsigned int getNumChildren() const;
  int getRank() const;
  double getTimeEst(const int, const int) const;
  double getBalTimeEst(const int, const int) const;
  int getWorkNeeded(const double) const;
  int getTotalWorkNeeded(const double, const int) const;
  unsigned int numTotalExternalBlockNeighbors();
  unsigned int numExternalBlockNeighbors();
  unsigned int numTotalSubDomains() const;
  SubDomain* getSubDomain(int index) const;
  SubDomain* globalGetSubDomain(int) const;
  SubDomain* getSubDomainLinear(int) const;
  SubDomain* popSubDomain() ;
  Node& getChild(int);
  const Node& getChild(int) const;
  unsigned int numSubDomains() const;
  double getWeight() const;
  double getEdgeWeight() const;
  double getTotalWeight(const int) const;
  double getMinEdgeWeight(const int) const;
};

void printNode(Node& node);
#endif
