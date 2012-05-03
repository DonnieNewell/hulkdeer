#include "Balancer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

Balancer::Balancer(){
}

Balancer::~Balancer(){

}

void Balancer::perfBalance(Cluster &cluster, Decomposition& decomp, int config)
{
	int num_blocks = decomp.getNumSubDomains();

	//perform initial task distribution
	this->balance(cluster, decomp, config);

	//get total iterations per second for cluster
	double total_weight = 0.0;
	for(int node=0; node<cluster.getNumNodes(); ++node)
	{
		Node& n = cluster.getNode(node);
		total_weight+= n.getWeight();
		for(int child=0; child < n.getNumChildren(); ++child)
		{
			total_weight += n.getChild(child).getWeight();
		}
	}

	//how long will it take with even distribution?
	double runtime_estimate = num_blocks / total_weight;

	fprintf(stderr, "perfBalance(): execution should take %f seconds.\n",runtime_estimate);	

	//balance the work
	Node &root = cluster.getNode(0);
	
}

void Balancer::balance(Cluster &cluster, Decomposition& decomp, int config)
{
    //num cpu nodes
	int total_nodes = cluster.getNumNodes();
	if(config > -1)
	{
		int num_children=0;
		for(int node = 0; node < cluster.getNumNodes(); ++node)
		{
			num_children += cluster.getNode(node).getNumChildren();
		}
		if(config == 1)
		{
			//gpu only
			total_nodes = num_children;
		}
		else if(config ==0)
		{
			//cpu and gpu
			total_nodes += num_children;
		}
	}

	int subd_per_node = ceil(decomp.getNumSubDomains()/(float)total_nodes);

	int task_index = 0;
	for(int node = 0; node < cluster.getNumNodes(); ++node)
	{
		Node& n = cluster.getNode(node);

		if(config <= 0){
				for(int subd=0; subd<subd_per_node && task_index < decomp.getNumSubDomains(); ++subd)
				{
						SubDomain3D &s = decomp.getSubDomain(task_index++);
						n.addSubDomain(s);
				}
		}
		if(config >= 0)
		{
			for(int device=0; device < n.getNumChildren(); ++device)
			{
				Node& child = n.getChild(device);
				for(int subd=0; subd<subd_per_node && task_index < decomp.getNumSubDomains(); ++subd)
				{
						SubDomain3D &s = decomp.getSubDomain(task_index++);
						child.addSubDomain(s);
				}
			}
			
		}


	}

}



void Balancer::balanceNode(Node& n, double runtime){
	WorkQueue wq;
	WorkRequest wr;

	for(int child=0; child<n.getNumChildren(); ++child)
	{
		Node& c = n.getChild(child);
		int diff = c.getTotalWorkNeeded(runtime)-c.numTotalSubDomains();
		if(0>diff) //child has extra work
		{
			int extra = abs(diff);
			for(int block=0; block<extra; ++block)
			{
				//move block from child to parent
				n.addSubDomain(c.removeSubDomain());
			}
		}
		else if(0<diff) //child needs more work
		{
			wr.setAmount(diff);
			wr.setIndex(child);
			wq.push(wr);
		}
	}

	/* 
		at this point we have all extra blocks, and 
		now we need to distribute blocks to children 
		that need it
	*/

	while(0 < n.numSubDomains() && //there are blocks left to give
		!wq.empty()) //there are requests left to fill
	{
		//get largest request
		WorkRequest tmp = wq.top();
		wq.pop();

		n.getChild(tmp.getIndex()).addSubDomain(n.removeSubDomain());

		//if there is still work left to do put it back on 
		// the queue so that it will reorder correctly
		tmp.setAmount(tmp.getAmount()-1);
		if(0 < tmp.getAmount())
			wq.push(tmp);
	}

	/*
		now we have balanced as much as we can and the only thing
		left is to get blocks from our parent, if we need them.
	*/
}
