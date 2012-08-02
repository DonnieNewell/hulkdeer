#include "./Balancer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

Balancer::Balancer() {
}

Balancer::~Balancer() {
}

void Balancer::perfBalanceGPU(Cluster &cluster, Decomposition& decomp) {
  const int kGPUOnly = 2;
  WorkQueue work_queue;
  WorkRequest work_request;
  if (decomp.getNumSubDomains() == 0 || cluster.getNumTotalGPUs() == 0)
    return;
  for (int gpu_index = 0; gpu_index < cluster.getNumTotalGPUs(); ++gpu_index) {
    Node& gpu = cluster.getGlobalGPU(gpu_index);
    // fastest gpu will have largest weight, and thus move to front of queue
    work_request.setTimeDiff(1.0 - gpu.getTimeEst(1, kGPUOnly));
    work_request.setIndex(gpu_index);
    work_queue.push(work_request);
  }

  // place data blocks on gpu's one-at-a-time
  while (decomp.getNumSubDomains() > 0) {
    work_request = work_queue.top();
    work_queue.pop();

    Node& gpu = cluster.getGlobalGPU(work_request.getIndex());
    SubDomain* block = decomp.popSubDomain();
    gpu.addSubDomain(block);

    work_request.setTimeDiff(1.0 - gpu.getTimeEst(1, kGPUOnly));
    work_queue.push(work_request);
  }
}

void Balancer::perfBalance(Cluster &cluster, Decomposition& decomp,
        const int kConfig) {
  WorkQueue work_queue;
  WorkRequest work_request;
  double total_weight(0.0);
  double min_edge_weight(0.0);
  double procTime(0.0);
  double commTime(0.0);
  double timeEst = procTime;
  bool changed(false);
  const int kGPUOnly(2);
  Node &root = cluster.getNode(0);
  size_t num_blocks = decomp.getNumSubDomains();

  // initialize block directory
  cluster.setNumBlocks(num_blocks);
  
  if (kGPUOnly == kConfig) {
    perfBalanceGPU(cluster, decomp);
  } else {
    // perform initial task distribution
    for (size_t i = 0; i < num_blocks; ++i)
      root.addSubDomain(decomp.popSubDomain());

    //get total iterations per second for cluster
    for (unsigned int node = 0; node < cluster.getNumNodes(); ++node) {
      total_weight += cluster.getNode(node).getTotalWeight(kConfig);
      min_edge_weight += cluster.getNode(node).getMinEdgeWeight(kConfig);
    }

    // quick estimation of runtime
    procTime = num_blocks / total_weight;
    commTime = num_blocks / min_edge_weight;
    timeEst = procTime + commTime;

    /*fprintf(stderr,
            "perfBalance: \n\ttime est: %f sec\n\tprocTime: %f sec\n\tcommTime: %f \
            sec\n\ttotal weight:%e \n\tmin edge weight:%e.\n",
            timeEst, procTime, commTime, total_weight, min_edge_weight);  // */

    do {
      changed = false;
      // balance the work between nodes and root
      for (unsigned int cpu_index = 1;
              cpu_index < cluster.getNumNodes();
              ++cpu_index) {
        Node& cpu_node = cluster.getNode(cpu_index);
        int work_deficit = cpu_node.getTotalWorkNeeded(timeEst, kConfig) - cpu_node.numTotalSubDomains();
        if (0 > work_deficit) { // node has extra work
          int extra_blocks = abs(work_deficit);
          for (int block_index = 0;
                  (block_index < extra_blocks) && (0 < cpu_node.numSubDomains());
                  ++block_index) {
            // move block from child to parent
            SubDomain* block = cpu_node.popSubDomain();
            if (NULL == block) {
              fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
            } else {
              root.addSubDomain(block);
              changed = true;
            }
          }
        } else if (0 < work_deficit) { //child needs more work
          work_request.setTimeDiff(timeEst - cpu_node.getTimeEst(0, kConfig));
          work_request.setIndex(cpu_index);
          work_queue.push(work_request);
        }
      }

      for (unsigned int cpu_index = 0;
              cpu_index < root.getNumChildren();
              ++cpu_index) {
        Node& cpu_node = root.getChild(cpu_index);
        int work_deficit = cpu_node.getTotalWorkNeeded(timeEst, kConfig) -
                cpu_node.numTotalSubDomains();
        if (0 > work_deficit) { // child has extra work
          int extra_blocks = abs(work_deficit);
          for (int block_index = 0;
                  (block_index < extra_blocks) && (0 < cpu_node.numSubDomains());
                  ++block_index) {
            // move block from child to parent
            SubDomain* block = cpu_node.popSubDomain();
            if (NULL == block) {
              fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
            } else {
              root.addSubDomain(block);
              changed = true;
            }
          }
        } else if (0 < work_deficit) { // child needs more work
          work_request.setTimeDiff(timeEst - cpu_node.getTimeEst(0, kConfig));
          work_request.setIndex(-1 * cpu_index); // hack so I know to give to one of root's children
          work_queue.push(work_request);
        }
      }
      /*
         at this point we have all extra blocks, and
         now we need to distribute blocks to children
         that need it
       */

      while (0 < root.numSubDomains() && // there are blocks left to give
              !work_queue.empty()) { // there are requests left to fill

        // get largest request
        WorkRequest tmp = work_queue.top();
        work_queue.pop();

        double newTimeDiff = 0.0;
        int id = tmp.getIndex();
        if (id <= 0) { // local child
          id = -1 * id;
          SubDomain* block = root.popSubDomain();
          if (NULL == block) {
            fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
          } else {
            root.getChild(id).addSubDomain(block);
            newTimeDiff = timeEst - root.getChild(id).getTimeEst(0, kConfig);
            changed = true;
          }
        } else { // request was from another node in cluster
          SubDomain* block = root.popSubDomain();
          if (NULL == block) {
            fprintf(stderr, "perfBalance: ERROR NULL subdomain pointer.\n");
          } else {
            cluster.getNode(id).addSubDomain(block);
            newTimeDiff = timeEst - cluster.getNode(id).getTimeEst(0, kConfig);
            changed = true;
          }
        }
        // if there is still work left to do put it back on
        // the queue so that it will reorder correctly
        if (0 < newTimeDiff) {
          tmp.setTimeDiff(newTimeDiff);
          work_queue.push(tmp);
        }
      }

      // balance the work within each node
      for (unsigned int node = 0; node < cluster.getNumNodes(); ++node) {
        changed |= balanceNode(cluster.getNode(node), timeEst, kConfig);
      }
    } while (changed);
  }
  /* the work is balanced, so we can fill the block directory */
  cluster.storeBlockLocs();
}

void Balancer::balance(Cluster &cluster, Decomposition& decomp,
        const int kConfig) {
  const int kCPUAndGPU = 0;
  const int kCPUOnly = 1;
  const int kGPUOnly = 2;
  int blocks_per_node = 0;
  // num cpu nodes
  unsigned int total_nodes = cluster.getNumNodes();
  size_t num_blocks = decomp.getNumSubDomains();

  // initialize block directory
  cluster.setNumBlocks(num_blocks);

  if (kConfig != kCPUOnly) {
    unsigned int num_gpus = 0;
    for (unsigned int node_index = 0;
            node_index < cluster.getNumNodes();
            ++node_index) {
      num_gpus += cluster.getNode(node_index).getNumChildren();
    }
    if (kConfig == kGPUOnly) // gpu only
      total_nodes = num_gpus;
    else if (kConfig == kCPUAndGPU) // cpu and gpu
      total_nodes += num_gpus;
  }

  blocks_per_node = ceil(decomp.getNumSubDomains() / (float) total_nodes);
  for (unsigned int node_index = 0;
          node_index < cluster.getNumNodes();
          ++node_index) {
    Node& node = cluster.getNode(node_index);
    if (kConfig == kCPUOnly || kConfig == kCPUAndGPU) {
      for (int subd = 0;
              subd < blocks_per_node && 0 < decomp.getNumSubDomains();
              ++subd) {
        SubDomain *block = decomp.popSubDomain();
        node.addSubDomain(block);
      }
    }
    if (kConfig == kGPUOnly || kConfig == kCPUAndGPU) {
      for (unsigned int gpu_index = 0;
              gpu_index < node.getNumChildren();
              ++gpu_index) {
        Node& gpu = node.getChild(gpu_index);
        for (int subd = 0;
                subd < blocks_per_node && 0 < decomp.getNumSubDomains();
                ++subd) {
          SubDomain *block = decomp.popSubDomain();
          gpu.addSubDomain(block);
        }
      }
    }
  }
  /* the work is balanced, so we can fill the block directory */
  cluster.storeBlockLocs();
}

bool Balancer::balanceNode(Node& node, double runtime, const int kConfig) {
  WorkQueue work_queue;
  WorkRequest work_request;
  bool changed = false;
  for (unsigned int gpu_index = 0;
          gpu_index < node.getNumChildren();
          ++gpu_index) {
    Node& gpu = node.getChild(gpu_index);
    int work_deficit = gpu.getTotalWorkNeeded(runtime, kConfig) -
            gpu.numTotalSubDomains();
    if (0 > work_deficit) { // child has extra work
      int extra_blocks = abs(work_deficit);
      for (int block = 0; block < extra_blocks; ++block) {
        // move block from child to parent
        node.addSubDomain(gpu.popSubDomain());
        changed = true;
      }
    } else if (0 < work_deficit) { // child needs more work
      work_request.setTimeDiff(runtime - gpu.getTimeEst(0, kConfig));
      work_request.setIndex(gpu_index);
      work_queue.push(work_request);
    }
  }

  /*
     at this point we have all extra blocks, and
     now we need to distribute blocks to children
     that need it
   */
  while (0 < node.numSubDomains() && //there are blocks left to give
          !work_queue.empty()) { //there are requests left to fill
    double time_diff = 0.0;
    //get largest request
    WorkRequest tmp = work_queue.top();
    work_queue.pop();
    node.getChild(tmp.getIndex()).addSubDomain(node.popSubDomain());
    time_diff = runtime - node.getChild(tmp.getIndex()).getTimeEst(0, kConfig);
    changed = true;

    // if there is still work left to do put it back on
    // the queue so that it will reorder correctly
    if (0 < time_diff) {
      tmp.setTimeDiff(time_diff);
      work_queue.push(tmp);
    }
  }

  /*
     now we have balanced as much as we can and the only thing
     left is to get blocks from our parent, if we need them.
   */
  return changed; //so we know to keep balancing
}
