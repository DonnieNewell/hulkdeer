#ifndef BALANCER_H
#define BALANCER_H

#include "Decomposition.h"
#include "Cluster.h"


class Balancer {
	private:
		void balanceNode(Node&);

	public:
		Balancer();
		~Balancer();
		void balance(Cluster&, Decomposition&, int);
        void perfBalanceGPU(Cluster&, Decomposition&);
		void perfBalance(Cluster&, Decomposition&, int);
		bool balanceNode(Node&, double, const int);

};


#endif
