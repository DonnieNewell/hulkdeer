#ifndef BALANCER_H
#define BALANCER_H

#include "Decomposition.h"
#include "Cluster.h"
class Balancer {
	private:

	public:
		Balancer();
		~Balancer();
		void balance(Cluster&,Decomposition&);

};
#endif
