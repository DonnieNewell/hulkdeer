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
		pair<int,int> req;
	public:
		WorkRequest()
		{
			req.first=0;
			req.second=0;
		}
		WorkRequest(int amount, int index)
		{
			req.first=amount;
			req.second=index;
		}

	bool operator<(const WorkRequest& rhs)const
	{
		return req.first < rhs.getAmount();
	}
	
	bool operator>(const WorkRequest& rhs)const
	{
		return req.first > rhs.getAmount();
	}
	int getAmount() const { return req.first; }
	int setAmount(int newAmount) { return req.first = newAmount; }
	int getIndex() const { return req.second; }
	int setIndex(int newIndex) { return req.second = newIndex; }
};

typedef priority_queue< WorkRequest, vector<WorkRequest>, greater<WorkRequest> > WorkQueue;
/************************************************/

class Node{
  double weight;
  int rank;
  vector<SubDomain3D> subD;
  vector<Node> children;
  WorkQueue wq;

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
    int getWorkNeeded(const double runtime) const;
    int getTotalWorkNeeded(const double runtime) const;
    const int numTotalSubDomains() const;
	SubDomain3D& getSubDomain(int index) ;	
	SubDomain3D removeSubDomain() ;	
    Node& getChild(int index) ;	
    const int numSubDomains() const;
    const double getWeight() const;
    const double getTotalWeight() const;
};

#endif
