#include "./Balancer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <boost/scoped_array.hpp>

Balancer::Balancer() {
}

Balancer::~Balancer() {
}

void Balancer::perfBalanceGPU(Cluster &cluster, Decomposition& decomp,
        const double kTimeEstimate) {
  const int kGPUOnly = 2;
  WorkQueue work_queue;
  WorkRequest work_request;
  const int kNumTotalGPUs = cluster.getNumTotalGPUs();
  if (decomp.getNumSubDomains() == 0 || kNumTotalGPUs == 0)
    return;
  for (int gpu_index = 0; gpu_index < kNumTotalGPUs; ++gpu_index) {
    Node& gpu = cluster.getGlobalGPU(gpu_index);
    // fastest gpu will have largest weight, and thus move to front of queue
    work_request.setTimeDiff(kTimeEstimate - gpu.getBalTimeEst(1, kGPUOnly));
    work_request.setIndex(gpu_index);
    work_queue.push(work_request);
  }

  const int kNumBlocks = decomp.getNumSubDomains();
  // place data blocks on gpu's one-at-a-time
  for (int block_id = 0; block_id < kNumBlocks; ++block_id) {
    work_request = work_queue.top();
    work_queue.pop();

    Node& gpu = cluster.getGlobalGPU(work_request.getIndex());
    gpu.incrementBalCount();

    double time_diff = gpu.getBalTimeEst(1, kGPUOnly);

    work_request.setTimeDiff(time_diff);
    work_queue.push(work_request);
    //printWorkQueue(work_queue);
  }

  cluster.distributeBlocks(&decomp);
}

void Balancer::populateAdjacencies(Cluster* cluster, int* adj_indices,
        int* adj_nodes, int* node_weights, int* adj_weights) {
    // validate
    if (cluster == NULL || adj_indices == NULL || adj_nodes == NULL) return;

    // go through each node and record the location of each neighbor
    const int kNumNodes = static_cast<int> (cluster->getNumNodes());
    const int kNumNeighbors3D = 26;
    const int kNumNeighbors2D = 8;

    int current_adj_index = 0;
    const int kNumDimensions = cluster->getBlockLinear(0)->getDimensionality();
    int num_neighbors = (kNumDimensions == 3) ?
                                kNumNeighbors3D :
                                kNumNeighbors2D;
    adj_indices[0] = current_adj_index;
    for (int node_index = 0; node_index < kNumNodes; ++node_index) {
        std::map<int, int> adj_map;
        Node& node = cluster->getNode(node_index);
        node_weights[node_index] = static_cast<int> (node.getWeight());
        const int kNumBlocks = static_cast<int> (node.numTotalSubDomains());
        for (int block_index = 0; block_index < kNumBlocks; ++block_index) {
            SubDomain* block = node.globalGetSubDomain(block_index);
            for (int neighbor_index = 0;
                    neighbor_index < num_neighbors;
                    ++neighbor_index) {
                const int kNoNeighbor = -1;
                const int kNodeRank = block->getNeighborLoc(neighbor_index);
                if (kNodeRank != kNoNeighbor) {  // block has a neighbor
                    adj_map[kNodeRank] = 1;
                }
            }
        }
        // transfer data from map to CSR adjacency array
        std::map<int, int>::iterator itr;
        for (itr = adj_map.begin();
                itr != adj_map.end();
                ++itr, ++current_adj_index) {
            adj_nodes[current_adj_index] = itr->first;  // key is node rank
            int edge_weight =
                static_cast<int> (cluster->getNode(itr->first).getEdgeWeight());
            adj_weights[current_adj_index] = edge_weight;
        }
        adj_indices[node_index + 1] = current_adj_index;
    }
}

void Balancer::minCut(Cluster &cluster) {
    const unsigned int kNumNodes = cluster.getNumNodes();
    // the last element in the indices array is the upper bound for
    //  the last list
    boost::scoped_array<int> adj_indices (new int[kNumNodes + 1]);
    boost::scoped_array<int> node_weights (new int[kNumNodes + 1]);
    // We store each edge 2 times and each node is connected to itself so n * n
    boost::scoped_array<int> adj_nodes (new int[kNumNodes * kNumNodes]);
    boost::scoped_array<int> adj_weights (new int[kNumNodes * kNumNodes]);
    populateAdjacencies(&cluster, adj_indices.get(), adj_nodes.get(),
            node_weights.get(), adj_weights.get());

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

  //get total iterations per second for cluster
  for (unsigned int node = 0; node < cluster.getNumNodes(); ++node) {
    total_weight += cluster.getNode(node).getTotalWeight(kConfig);
    min_edge_weight += cluster.getNode(node).getMinEdgeWeight(kConfig);
  }

  // quick estimation of runtime
  procTime = num_blocks / total_weight;
  commTime = num_blocks / min_edge_weight;
  timeEst = procTime + commTime;

  if (kGPUOnly == kConfig) {
    perfBalanceGPU(cluster, decomp, timeEst);
  } else {
    // perform initial task distribution
    for (size_t i = 0; i < num_blocks; ++i)
      root.incrementBalCount();

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
        int work_deficit = cpu_node.getTotalWorkNeeded(timeEst, kConfig) -
                                cpu_node.getBalCount();
        if (0 > work_deficit) { // node has extra work
          int extra_blocks = abs(work_deficit);
          for (int block_index = 0;
                  (block_index < extra_blocks) &&
                  (0 < cpu_node.getBalCount());
                  ++block_index) {
            // move block from child to parent
            cpu_node.decrementBalCount();
            root.incrementBalCount();
            changed = true;
          }
        } else if (0 < work_deficit) { //child needs more work
          work_request.setTimeDiff(timeEst - cpu_node.getBalTimeEst(0, kConfig));
          work_request.setIndex(cpu_index);
          work_queue.push(work_request);
        }
      }

      for (unsigned int cpu_index = 0;
              cpu_index < root.getNumChildren();
              ++cpu_index) {
        Node& cpu_node = root.getChild(cpu_index);
        int work_deficit = cpu_node.getTotalWorkNeeded(timeEst, kConfig) -
                cpu_node.getBalCount();
        if (0 > work_deficit) { // child has extra work
          int extra_blocks = abs(work_deficit);
          for (int block_index = 0;
                  (block_index < extra_blocks) && (0 < cpu_node.getBalCount());
                  ++block_index) {
            // move block from child to parent
            cpu_node.decrementBalCount();
            root.incrementBalCount();
            changed = true;
          }
        } else if (0 < work_deficit) { // child needs more work
          work_request.setTimeDiff(timeEst - cpu_node.getBalTimeEst(0, kConfig));
          work_request.setIndex(-1 * cpu_index); // hack so I know to give to one of root's children
          work_queue.push(work_request);
        }
      }
      /*
         at this point we have all extra blocks, and
         now we need to distribute blocks to children
         that need it
       */

      while (0 < root.getBalCount() && // there are blocks left to give
              !work_queue.empty()) { // there are requests left to fill

        // get largest request
        WorkRequest tmp = work_queue.top();
        work_queue.pop();

        double newTimeDiff = 0.0;
        int id = tmp.getIndex();
        if (id <= 0) { // local child
          id = -1 * id;
          root.decrementBalCount();
            root.getChild(id).incrementBalCount();
            newTimeDiff = timeEst - root.getChild(id).getBalTimeEst(0, kConfig);
            changed = true;
        } else { // request was from another node in cluster
          root.decrementBalCount();
            cluster.getNode(id).incrementBalCount();
            newTimeDiff = timeEst - cluster.getNode(id).getBalTimeEst(0, kConfig);
            changed = true;
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

  /* now that we know where everything should go, distribute the blocks */
  cluster.distributeBlocks(&decomp);

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
            gpu.getBalCount();
    if (0 > work_deficit) { // child has extra work
      int extra_blocks = abs(work_deficit);
      for (int block = 0; block < extra_blocks; ++block) {
        // move block from child to parent
        gpu.decrementBalCount();
        node.incrementBalCount();
        changed = true;
      }
    } else if (0 < work_deficit) {  // child needs more work
      work_request.setTimeDiff(runtime - gpu.getBalTimeEst(0, kConfig));
      work_request.setIndex(gpu_index);
      work_queue.push(work_request);
    }
  }

  /*
     at this point we have all extra blocks, and
     now we need to distribute blocks to children
     that need it
   */
  while (0 < node.getBalCount() &&  // there are blocks left to give
          !work_queue.empty()) {  // there are requests left to fill
    double time_diff = 0.0;
    //get largest request
    WorkRequest tmp = work_queue.top();
    work_queue.pop();
    node.decrementBalCount();
    node.getChild(tmp.getIndex()).incrementBalCount();
    time_diff = runtime - node.getChild(tmp.getIndex()).getBalTimeEst(0, kConfig);
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
