#include "Balancer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

Balancer::Balancer () { }

Balancer::~Balancer () { }

void Balancer::perfBalance (Cluster &cluster, Decomposition& decomp, int config) {
  WorkQueue wq;
  WorkRequest wr;
  Node &root = cluster.getNode(0);

  size_t num_blocks = decomp.getNumSubDomains();

  //initialize block directory
  cluster.setNumBlocks(num_blocks);

  //perform initial task distribution
  // this->balance(cluster, decomp, config);
  for (size_t i=0; i<num_blocks; ++i) {
    root.addSubDomain(decomp.popSubDomain());
  }

  //get total iterations per second for cluster
  double total_weight = 0.0;
  double min_edge_weight = 0.0;
  for (unsigned int node=0; node < cluster.getNumNodes(); ++node) {
    total_weight    += cluster.getNode(node).getTotalWeight()   ;
    min_edge_weight += cluster.getNode(node).getMinEdgeWeight() ;
  }

  //quick estimation of runtime
  double procTime = num_blocks / total_weight     ;
  double commTime = num_blocks / min_edge_weight  ;
  double timeEst  = procTime + commTime * commTime;

  fprintf(stderr,
          "perfBalance: \n\ttime est: %f sec\n\ttotal weight:%e \
          \n\tmin edge weight:%e.\n",
          timeEst, total_weight, min_edge_weight);

  bool changed;

  /*
     TODO
      need to change the way that work requests are created because
     they now need to store the estimated runtime instead of amount
     requested.
   */
  int counter = 0; //DEBUG purposes
  do {
    changed = false;
    counter++      ;
    //balance the work between nodes and root
    for (unsigned int child=1; child<cluster.getNumNodes(); ++child) {
      Node& c    = cluster.getNode(child);
      int   diff = c.getTotalWorkNeeded(timeEst)-c.numTotalSubDomains();
      if (0>diff) {//child has extra work

        int extra = abs(diff);
        for (int block=0;
            (block < extra) && (0 < c.numSubDomains());
            ++block) {
          //move block from child to parent
          SubDomain* s = c.popSubDomain();
          if (NULL == s) {
            fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
          } else {
            root.addSubDomain(s);
            changed = true;
          }
        }
      } else if (0<diff) {  //child needs more work
        wr.setTimeDiff(timeEst-c.getTimeEst(0));
        wr.setIndex(child);
        wq.push(wr);
      }
    }

    for (unsigned int child=0; child < root.getNumChildren(); ++child) {
      Node& c = root.getChild(child);
      int diff = c.getTotalWorkNeeded(timeEst)-c.numTotalSubDomains();
      if (0>diff) {//child has extra work
        int extra = abs(diff);
        for (int block=0; (block<extra)&&(0<c.numSubDomains()); ++block) {
          //move block from child to parent
          SubDomain* s = c.popSubDomain();
          if (NULL == s) {
            fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
          } else {
            root.addSubDomain(s);
            changed = true;
          }
        }
      } else if (0<diff) {//child needs more work

        wr.setTimeDiff(timeEst - c.getTimeEst(0));
        wr.setIndex(-1*child);//hack so I know to give to one of root's children
        wq.push(wr);
      }
    }
    /*
       at this point we have all extra blocks, and
       now we need to distribute blocks to children
       that need it
     */

    while (0 < root.numSubDomains() && //there are blocks left to give
          !wq.empty()) {//there are requests left to fill

      //get largest request
      WorkRequest tmp = wq.top();
      wq.pop();

      double newTimeDiff = 0.0;
      int id = tmp.getIndex();
      if (id<=0) {//local child
        id = -1*id;
        SubDomain* s = root.popSubDomain();
        if (NULL == s) {
          fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
        } else {
          root.getChild(id).addSubDomain(s);
          newTimeDiff = timeEst - root.getChild(id).getTimeEst(0);
          changed=true;
        }
      } else { //request was from another node in cluster
        SubDomain* s = root.popSubDomain();
        if(NULL == s){
          fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
        } else {
          cluster.getNode(id).addSubDomain(s);
          newTimeDiff = timeEst - cluster.getNode(id).getTimeEst(0);
          changed=true;
        }
      }
      //if there is still work left to do put it back on
      // the queue so that it will reorder correctly
      if (0 < newTimeDiff) {
        tmp.setTimeDiff(newTimeDiff);
        wq.push(tmp);
      }
    }


    //balance the work within each node
    for (unsigned int node=0; node < cluster.getNumNodes(); ++node) {
      changed |= balanceNode(cluster.getNode(node),timeEst);
    }
  } while (changed);

  /* the work is balanced, so we can fill the block directory */
  cluster.storeBlockLocs();
  printNeighbors(cluster.getNode(0).getSubDomain(0)); /* DEBUG */
}

void Balancer::balance (Cluster &cluster, Decomposition& decomp, int config) {
  //num cpu nodes
  unsigned int total_nodes = cluster.getNumNodes();
  if (config > -1) {
    unsigned int num_children = 0;
    for (unsigned int node = 0; node < cluster.getNumNodes(); ++node) {
      num_children += cluster.getNode(node).getNumChildren();
    }
    if (config == 1) {
      //gpu only
      total_nodes = num_children;
    } else if (config == 0) {
      //cpu and gpu
      total_nodes += num_children;
    }
  }
  int subd_per_node = ceil(decomp.getNumSubDomains()/(float)total_nodes);
  for (unsigned int node = 0; node < cluster.getNumNodes(); ++node) {
    Node& n = cluster.getNode(node);
    if (config <= 0) {
      for (int subd=0;
            subd < subd_per_node && 0 < decomp.getNumSubDomains();
            ++subd) {
        SubDomain *s = decomp.popSubDomain();
        n.addSubDomain(s);
      }
    }
    if (config >= 0) {
      for (unsigned int device = 0; device < n.getNumChildren(); ++device) {
        Node& child = n.getChild(device);
        for (int subd = 0;
              subd < subd_per_node && 0 < decomp.getNumSubDomains();
              ++subd) {
          SubDomain *s = decomp.popSubDomain();
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
  for (unsigned int child=0; child < n.getNumChildren(); ++child) {
    Node& c = n.getChild(child);
    int diff = c.getTotalWorkNeeded(runtime)-c.numTotalSubDomains();
    if (0 > diff) {  //child has extra work
      int extra = abs(diff);
      for (int block=0; block < extra; ++block) {
        //move block from child to parent
        n.addSubDomain(c.popSubDomain());
        changed = true;
      }
    } else if (0<diff) {  //child needs more work
      wr.setTimeDiff(runtime - c.getTimeEst(0));
      wr.setIndex(child);
      wq.push(wr);
    }
  }

  /*
     at this point we have all extra blocks, and
     now we need to distribute blocks to children
     that need it
   */
  while (0 < n.numSubDomains() &&  //there are blocks left to give
          !wq.empty()) {  //there are requests left to fill
    double timeDiff = 0.0;
    //get largest request
    WorkRequest tmp = wq.top();
    wq.pop();
    n.getChild(tmp.getIndex()).addSubDomain(n.popSubDomain());
    timeDiff = runtime - n.getChild(tmp.getIndex()).getTimeEst(0);
    changed=true;

    //if there is still work left to do put it back on
    // the queue so that it will reorder correctly
    if (0 < timeDiff) {
      tmp.setTimeDiff(timeDiff);
      wq.push(tmp);
    }
  }

  /*
     now we have balanced as much as we can and the only thing
     left is to get blocks from our parent, if we need them.
   */
  return changed; //so we know to keep balancing
}
