#ifndef NODE_H 
#define NODE_H
#include "SubDomain3D.h"
#include <vector>
#include <queue>
#include <utility>

using namespace std;

class WorkRequest
{
  private:
    pair<double,int> req;
  public:
    WorkRequest()
    {
      req.first  = 0.0;
      req.second = 0  ;
    }
    WorkRequest(const WorkRequest& rhs)
    {
      if(this != &rhs)
      {
        req.first   = rhs.getTimeDiff();
        req.second  = rhs.getIndex()   ;
      }
    }
    WorkRequest(double timeDiff, int index )
    {
      req.first  = timeDiff;
      req.second = index   ;
    }

    ~WorkRequest(){ }
    bool operator<(const WorkRequest& rhs)const
    {
      return req.first < rhs.getTimeDiff();
    }

    bool operator>(const WorkRequest& rhs)const
    {
      return req.first > rhs.getTimeDiff();
    }
    double getTimeDiff() const { return req.first; }
    double setTimeDiff(double newTimeDiff) { return req.first = newTimeDiff; }
    int getIndex() const { return req.second; }
    int setIndex(int newIndex) { return req.second = newIndex; }
};

typedef priority_queue< WorkRequest, vector<WorkRequest>, greater<WorkRequest> > WorkQueue;
/************************************************/

class Node{
  double weight;
  double edgeWeight;
  int rank;
  vector<SubDomain3D*> subD;
  vector<Node> children;

  public: 
  Node();
  Node(double);
  Node(const Node&);
  ~Node();
  Node& operator=(const Node& rhs);
  void addSubDomain(SubDomain3D*);	
  void setEdgeWeight(double);
  void setWeight(double);
  void setRank(int);
  void setNumChildren(int);
  const int getNumChildren() const;
  const int getRank() const;
  const double getTimeEst(int extra) const;
  int getWorkNeeded(const double runtime) const;
  int getTotalWorkNeeded(const double runtime) const;
  const int numTotalSubDomains() const;
  SubDomain3D* getSubDomain(int index) ;	
  SubDomain3D* popSubDomain() ;	
  Node& getChild(int index) ;	
  const int numSubDomains() const;
  const double getWeight() const;
  const double getEdgeWeight() const;
  const double getTotalWeight() const;
  const double getMinEdgeWeight() const;
};

#endif
