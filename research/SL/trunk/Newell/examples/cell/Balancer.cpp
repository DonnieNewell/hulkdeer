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
  WorkQueue wq;
  WorkRequest wr;

  int num_blocks = decomp.getNumSubDomains();

  //perform initial task distribution
  this->balance(cluster, decomp, config);

  //get total iterations per second for cluster
  double total_weight = 0.0;
  for(int node=0; node<cluster.getNumNodes(); ++node)
  {
    total_weight += cluster.getNode(node).getTotalWeight();
  }

  //how long will it take with ideal distribution?
  double runtime_estimate = num_blocks / total_weight;

  fprintf(stderr, "perfBalance(): execution should take %f seconds.\n",runtime_estimate);	

  bool changed;

  Node &root = cluster.getNode(0);
  int counter = 0; //DEBUG purposes
  do{
    changed=false;
    counter++;
    //balance the work between nodes and root
    for(int child=1; child<cluster.getNumNodes(); ++child)
    {
      Node& c = cluster.getNode(child);
      int diff = c.getTotalWorkNeeded(runtime_estimate)-c.numTotalSubDomains();
      if(0>diff) //child has extra work
      {
        int extra = abs(diff);
        for(int block=0; (block<extra) && (0 < c.numSubDomains()); ++block)
        {
          //move block from child to parent
          SubDomain3D* s = c.popSubDomain();
          if(NULL == s)
          {
            fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
          }
          else
          {
            root.addSubDomain(s);
            changed = true;
          }
        }
      }
      else if(0<diff) //child needs more work
      {
        wr.setAmount(diff);
        wr.setIndex(child);
        wq.push(wr);
      }
    }

    for(int child=0; child<root.getNumChildren(); ++child)
    {
      Node& c = root.getChild(child);
      int diff = c.getTotalWorkNeeded(runtime_estimate)-c.numTotalSubDomains();
      if(0>diff) //child has extra work
      {
        int extra = abs(diff);
        for(int block=0; (block<extra)&&(0<c.numSubDomains()); ++block)
        {
          //move block from child to parent
          SubDomain3D* s = c.popSubDomain();
          if(NULL == s)
          {
            fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
          }
          else
          {
            root.addSubDomain(s);
            changed = true;
          }
        }
      }
      else if(0<diff) //child needs more work
      {
        wr.setAmount(diff);
        wr.setIndex(-1*child);//hack so I know to give to one of root's children
        wq.push(wr);
      }
    }
    /* 
       at this point we have all extra blocks, and 
       now we need to distribute blocks to children 
       that need it
     */

    while(0 < root.numSubDomains() && //there are blocks left to give
        !wq.empty()) //there are requests left to fill
    {
      //get largest request
      WorkRequest tmp = wq.top();
      wq.pop();

      int id = tmp.getIndex();
      if(id<=0) //local child
      {
        id = -1*id;
        SubDomain3D* s = root.popSubDomain();
        if(NULL == s)
        {
          fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
        }
        else
        {

          root.getChild(id).addSubDomain(s);
          changed=true;
        }
      }
      else //request was from another node in cluster
      {
        SubDomain3D* s = root.popSubDomain();
        if(NULL == s)
        {
          fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
        }
        else
        {

          cluster.getNode(id).addSubDomain(s);
          changed=true;
        }
      }

      //if there is still work left to do put it back on 
      // the queue so that it will reorder correctly
      tmp.setAmount(tmp.getAmount()-1);
      if(0 < tmp.getAmount())
        wq.push(tmp);
    }


    //balance the work within each node
    for(int node=0; node<cluster.getNumNodes(); ++node)
    {
      changed |= balanceNode(cluster.getNode(node),runtime_estimate);
    }
  }while(changed);

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


  for(int node = 0; node < cluster.getNumNodes(); ++node)
  {
    Node& n = cluster.getNode(node);

    if(config <= 0){
      for(int subd=0; subd<subd_per_node && 0 < decomp.getNumSubDomains(); ++subd)
      {
        SubDomain3D *s = decomp.popSubDomain();
        n.addSubDomain(s);
      }
    }
    if(config >= 0)
    {
      for(int device=0; device < n.getNumChildren(); ++device)
      {
        Node& child = n.getChild(device);
        for(int subd=0; subd<subd_per_node && 0 < decomp.getNumSubDomains(); ++subd)
        {
          SubDomain3D *s = decomp.popSubDomain();
          child.addSubDomain(s);
        }
      }

    }

  }

}



bool Balancer::balanceNode(Node& n, double runtime){
  WorkQueue wq;
  WorkRequest wr;
  bool changed=false;

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
        n.addSubDomain(c.popSubDomain());
        changed = true;
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

    n.getChild(tmp.getIndex()).addSubDomain(n.popSubDomain());
    changed=true;

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
  return changed; //so we know to keep balancing
}
